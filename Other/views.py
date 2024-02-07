from django.shortcuts import render, redirect
from django.urls import reverse
from django.views.generic import CreateView, DetailView
from django.http import HttpResponseRedirect, HttpResponse
from django.contrib import messages
from django.contrib.auth.decorators import login_required, permission_required
from django.contrib.auth import authenticate, login, logout
from django.core.exceptions import PermissionDenied
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework.permissions import IsAuthenticated
from django.contrib import auth
from datetime import datetime, timedelta
from PIL import Image
import asyncio, logging, time, datetime
from asgiref.sync import sync_to_async, async_to_sync
from celery import shared_task
import json , re, numpy as np
from .forms import *
from .models import *
from fetch_data.utilities.tools import territory_divider, xyz2bbox_territory, coords_2_xyz_newton, get_current_datetime, territory_tags
from fetch_data.utilities.image_db import *
from .serializers import *

fetch_chunk_size=8
concat_size_limit = 10000


# @shared_task
def fetch(x_range, y_range, zoom, start_date, end_date, overwrite_repetitious, inference, save_concated, lon_min, lat_min, lon_max, lat_max, user, confidence_threshold=0.9):
    global fetch_chunk_size
    (x_min, x_max), (y_min, y_max) = x_range, y_range
    parent_total_queries = (x_max - x_min + 1) * (y_max - y_min + 1)
    parent_queries_done = 0

    territories = territory_divider(x_range, y_range, piece_size=fetch_chunk_size, flattened=True)
    print(f"\n\n\nterritories: ", territories, "\n\n\n")
    subtasks = False if len(territories) == 1 else True
    logging.info(f"\nterritories:\n{territories}\n")

    last_internal_variables = InternalVariables.objects.first()
    if last_internal_variables == None:
        last_internal_variables = InternalVariables.objects.create(last_task_id=0)
    last_task_id = last_internal_variables.last_task_id
    parent_task_id = str(int(last_task_id) + 1)
    last_internal_variables.last_task_id = int(parent_task_id)
    last_internal_variables.save()

    # parent_task_id = f"{user.username}-[{lon_min},{lat_min},{lon_max},{lat_max}]-({start_date}_{end_date})-q_{get_current_datetime()}"
    task_ids = [f"{parent_task_id}_part{i+1}" for i in range(len(territories))]
    task_type = "fetch_infer" if inference else "fetch"

    parent_task = QueuedTask.objects.create(task_id=parent_task_id, is_parent=True, task_type=task_type, task_status="fetching", fetch_progress=0, 
                           lon_min=lon_min, lat_min=lat_min, lon_max=lon_max, lat_max=lat_max, 
                           zoom=zoom, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max,
                           time_from=start_date, time_to=end_date, user_queued=user, confidence_threshold=confidence_threshold)
    
    territory_coords_parent = (lon_min, lat_min, lon_max, lat_max)
    area_tags_parent = territory_tags(territory_coords_parent, margin_neglect=0.01)
    for tag in area_tags_parent:
        area = PresetArea.objects.get(tag=tag)
        parent_task.area_tag.add(area)
        parent_task.save()
    parent_task.save()

    if subtasks:
        for idx, territory in enumerate(territories):
            x_range_child = territory[0]
            y_range_child = territory[1]
            (x_min_child, x_max_child), (y_min_child, y_max_child) = x_range_child, y_range_child
            territory_coords_child = xyz2bbox_territory(x_range_child, y_range_child, zoom)
            lon_min_child, lat_min_child, lon_max_child, lat_max_child = territory_coords_child

            child_task_id = task_ids[idx]
            child_task = QueuedTask.objects.create(task_id=child_task_id, is_parent=False, task_type=task_type, task_status="fetching",
                                                fetch_progress=0, lon_min=lon_min_child, lat_min=lat_min_child, lon_max=lon_max_child,
                                                lat_max=lat_max_child, zoom=zoom, x_min=x_min_child, x_max=x_max_child, y_min=y_min_child, y_max=y_max_child,
                                                time_from=start_date, time_to=end_date, user_queued=user, confidence_threshold=confidence_threshold)
            parent_task.child_task.add(child_task)
            parent_task.save()
            area_tags_child = territory_tags(territory_coords_child, margin_neglect=0.01)
            for tag in area_tags_child:
                area = PresetArea.objects.get(tag=tag)
                child_task.area_tag.add(area)
            child_task.save()
        for idx, territory in enumerate(territories):
            logging.info(f"Territory {idx} (out of {len(territories)}) began fetching")
            x_range_child = territory[0]
            y_range_child = territory[1]
            territory_coords_child = xyz2bbox_territory(x_range, y_range, zoom)
            lon_min_child, lat_min_child, lon_max_child, lat_max_child = territory_coords_child

            child_task_id = task_ids[idx]
            t1 = time.perf_counter()
            parent_queries_done = territory_fetch_inference(x_range_child, y_range_child, zoom, start_date=start_date, end_date=end_date, child_task_id=child_task_id,
                                    parent_task_id=parent_task_id, subtasks=True ,parent_queries_done=parent_queries_done, parent_total_queries=parent_total_queries,
                                    overwrite_repetitious=overwrite_repetitious, inference=inference, save_concated=save_concated, confidence_threshold=confidence_threshold)
            logging.info(f"{(time.perf_counter() - t1):.0f} seconds elapsed to fetch this territory")
        
        parent_task.task_status = "inferenced"
        parent_task.fetch_progress = 100
        parent_task.save()
    else:
        territory_fetch_inference(x_range, y_range, zoom, start_date=start_date, end_date=end_date, child_task_id=parent_task_id, confidence_threshold=confidence_threshold,
                                  parent_task_id= None, subtasks=False ,parent_queries_done=None, parent_total_queries=None, overwrite_repetitious=overwrite_repetitious,
                                  inference=inference, save_concated=save_concated)

    return


@login_required(login_url='users:login')
def territory_fetch(request):
    # user = await sync_to_async(auth.get_user)(request)
    user = request.user
    if request.method == 'GET':
        form = SentinelFetchForm(initial={"confidence_threshold": 90})
        return render(request, "fetch_data/SentinelFetch.html", context={'preset_araes': PresetArea.objects.all(),'form': form, 'user': user})

    elif request.method == 'POST' and 'fetch' in request.POST and request.user.is_authenticated:
        form = SentinelFetchForm(request.POST)
        coordinate_type = request.POST.get('coordinate_type')
        coordinate_type = "lonlat" if coordinate_type == None else coordinate_type
        date_type = request.POST.get('date_type')
        save_concated = True if request.POST.get('save_concated') == "True" else False
        if form.is_valid():
            zoom = form.cleaned_data['zoom']
            if zoom is None:
                message = 'Please specify a value for "Zoom". Downloading and storing tile images will be performed based on this value; be informed the default Zoom is 14.'
                return render(request, "fetch_data/SentinelFetch.html", {'form': form, 'user':request.user, "error": "other", "message": message})
            zoom = int(zoom)
            if coordinate_type == "xy":
                x_min = form.cleaned_data['x_min']
                x_max = form.cleaned_data['x_max']
                y_min = form.cleaned_data['y_min']
                y_max = form.cleaned_data['y_max']
                x_range = [int(x_min), int(x_max)]
                y_range = [int(y_min), int(y_max)]
                lon_min, lat_min, lon_max, lat_max = xyz2bbox_territory(x_range, y_range, zoom)
            elif coordinate_type == "lonlat":
                lon_min = form.cleaned_data['lon_min']
                lon_max = form.cleaned_data['lon_max']
                lat_min = form.cleaned_data['lat_min']
                lat_max = form.cleaned_data['lat_max']
                coords = (lon_min, lat_min, lon_max, lat_max)
                if all([lon_min, lat_min, lon_max, lat_max]) is False:
                    message = 'All terms of coordinates (Lon min, Lat min, Lon max, Lat max) should be given! Please modify your inputs.'
                    return render(request, "fetch_data/SentinelFetch.html", {'form': form, 'user':request.user, "error": "other", "message": message}) 
                x_range, y_range, _ = coords_2_xyz_newton(coords, zoom)
                (x_min, x_max), (y_min, y_max) = x_range, y_range
            if date_type == 'start_end':
                start_date = form.cleaned_data['start_date']
                end_date = form.cleaned_data['end_date']
            elif date_type == 'days_before':
                n_days_before_base_date = form.cleaned_data['n_days_before_base_date']
                base_date = form.cleaned_data['base_date']
                date_data = start_end_time_interpreter(n_days_before_base_date=n_days_before_base_date, base_date=base_date)
                start_date = date_data['start_date']
                end_date = date_data['end_date']

            overwrite_repetitious = form.cleaned_data['overwrite_repetitious']
            inference = form.cleaned_data['inference']
            confidence_threshold = float(int(form.cleaned_data['confidence_threshold']) / 100)
            if start_date > end_date:
                message = 'Start date is greater than end date. Please modify your inputs.'
                return render(request, "fetch_data/SentinelFetch.html", {'form': form, 'user':request.user, "error": "other", "message": message})
            if (start_date is None) or (end_date is None):
                message = 'Start date and end date both should be specified. Please modify your inputs.'
                return render(request, "fetch_data/SentinelFetch.html", {'form': form, 'user':request.user, "error": "other", "message": message})
            if start_date > datetime.date.today():
                message = 'Start date cannot be a date in future! Please modify your inputs.'
                return render(request, "fetch_data/SentinelFetch.html", {'form': form, 'user':request.user, "error": "other", "message": message})
            if end_date > datetime.date.today():
                message = 'End date cannot be a date in future! Please modify your inputs.'
                return render(request, "fetch_data/SentinelFetch.html", {'form': form, 'user':request.user, "error": "other", "message": message}) 
            if all([lon_min, lat_min, lon_max, lat_max]) is False:
                message = 'All terms of coordinates (Lon min, Lat min, Lon max, Lat max) should be given! Please modify your inputs.'
                return render(request, "fetch_data/SentinelFetch.html", {'form': form, 'user':request.user, "error": "other", "message": message}) 
                
            
            total_tiles_limit = 40000
            total_tiles = (x_max - x_min + 1) * (y_max - y_min + 1)
            if total_tiles > total_tiles_limit:
                # message = f"The area is excessively large, comprising a total of {total_tiles} tile images. Fetch requests exceeding a threshold of {total_tiles_limit} tiles will not be accepted, as this would impose significant strain on network resources and processing capabilities."
                return render(request, "fetch_data/SentinelFetch.html", {'form': form, 'user':request.user, "error": "large_area", "total_tiles": total_tiles, "total_tiles_limit": total_tiles_limit})

            # territories = territory_divider(x_range, y_range, piece_size=70)
            # logging.info(f"\nterritories:\n{territories}\n")

            # QueuedTask lines
            # task_id_base = f"{user.username}-[{lon_min},{lat_min},{lon_max},{lat_max}]-({start_date}_{end_date})-q_{get_current_datetime()}"
            # task_ids = [f"{task_id_base}_part{i+1}" for i in range(len(territories))]
            # task_type = "fetch_infer" if inference else "fetch"
            # try:
            #     area_tag = PresetArea.objects.get(lon_min=lon_min, lat_min=lat_min, lon_max=lon_max, lat_max=lat_max)
            # except PresetArea.DoesNotExist:
            #     area_tag = None
            # task = QueuedTask.objects.create(task_id=task_id, task_type=task_type, task_status="fetching", fetch_progress=0, 
            #                            lon_min=lon_min, lat_min=lat_min, lon_max=lon_max, lat_max=lat_max, 
            #                            zoom=zoom, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max,
            #                            time_from=start_date, time_to=end_date, user_queued=request.user,)
            # if area_tag != None:
            #     task.area_tag=area_tag
            #     task.save()
            ### End QueuedTask lines
            # fetch(territories, task_ids, task_type, x_min, x_max, y_min, y_max, zoom, start_date, end_date,
            #       overwrite_repetitious, inference, save_concated, lon_min=lon_min, lat_min=lat_min, lon_max=lon_max, lat_max=lat_max,)
            # t0 = time.perf_counter()
            confidence_threshold = 0.9
            fetch(x_range, y_range, zoom, start_date, end_date, overwrite_repetitious, inference, save_concated,
                  lon_min=lon_min, lat_min=lat_min, lon_max=lon_max, lat_max=lat_max, user=user, confidence_threshold=confidence_threshold)

            # loop = asyncio.get_event_loop()
            # async_task = loop.create_task(fetch(territories, x_min, x_max, y_min, y_max, zoom, start_date, end_date, task_id, overwrite_repetitious, inference, save_concated))
            # await async_task
            
            # fetch.delay(territories, x_min, x_max, y_min, y_max, zoom, start_date, end_date, task_id,
                        # overwrite_repetitious, inference, save_concated)
            
            # fetch(territories, x_min, x_max, y_min, y_max, zoom, start_date, end_date, task_id,
            # overwrite_repetitious, inference, save_concated)
            # logging.info(f"{(time.perf_counter() - t0):.0f} seconds elapsed to fetch all territories")
            logging.info(f"All territories fetched")
            return render(request, "fetch_data/success.html", context={"form_cleaned_data": form.cleaned_data})
        else:
            return render(request, "fetch_data/error.html", context={'errors': form.errors})
    
    elif request.method == 'POST' and 'fetch' in request.POST and request.user.is_authenticated == False:
        message = "Please login first"
        return HttpResponseRedirect(reverse('users:login'))
    
    elif request.method == 'POST' and 'fill_coords' in request.POST:
        form = SentinelFetchForm(request.POST)
        if form.is_valid():
            preset_area_id = form.cleaned_data['preset_area']
            zoom = int(form.cleaned_data['zoom'])
            preset_area = PresetArea.objects.get(tag=preset_area_id)
            coords = [preset_area.lon_min, preset_area.lat_min, preset_area.lon_max, preset_area.lat_max]
            lon_min, lat_min, lon_max, lat_max = coords
            if zoom == 14:
                (x_min, x_max), (y_min, y_max) = preset_area.x_range_z14(), preset_area.y_range_z14()
            else:
                x_range, y_range, _ = coords_2_xyz_newton(coords, zoom)
                (x_min, x_max), (y_min, y_max) = x_range, y_range

            start_date = form.cleaned_data['start_date']
            end_date = form.cleaned_data['end_date']
            base_date = form.cleaned_data['base_date']
            n_days_before_base_date = form.cleaned_data['n_days_before_base_date']
            inference = form.cleaned_data['inference']
            overwrite_repetitious = form.cleaned_data['overwrite_repetitious']
            save_concated = True if request.POST.get('save_concated') == "True" else False
            confidence_threshold = form.cleaned_data['confidence_threshold']
            form = SentinelFetchForm(initial={"x_min": x_min, "x_max": x_max, "y_min": y_min,
                                              "y_max": y_max,"zoom": zoom, "lon_min": lon_min,
                                              "lon_max": lon_max, "lat_min": lat_min,
                                              "lat_max": lat_max, "start_date": start_date, "end_date": end_date,
                                              "base_date": base_date, "n_days_before_base_date": n_days_before_base_date,
                                              "overwrite_repetitious": overwrite_repetitious, "preset_area": preset_area_id,
                                              'inference': inference, "confidence_threshold": confidence_threshold,
                                               })
            return render(request, "fetch_data/SentinelFetch.html", {'form': form, 'user':request.user})
        
    elif request.method == 'POST' and 'last_10_days' in request.POST:
        form = SentinelFetchForm(request.POST)
        if form.is_valid():
            preset_area = form.cleaned_data['preset_area']
            zoom = form.cleaned_data['zoom']
            lon_min = form.cleaned_data['lon_min']
            lon_max = form.cleaned_data['lon_max']
            lat_min = form.cleaned_data['lat_min']
            lat_max = form.cleaned_data['lat_max']
            x_min = form.cleaned_data['x_min']
            x_max = form.cleaned_data['x_max']
            y_min = form.cleaned_data['y_min']
            y_max = form.cleaned_data['y_max']

            end_date = datetime.datetime.now().date()
            start_date = end_date - datetime.timedelta(days=10)
            base_date = end_date
            n_days_before_base_date = 10
            inference = form.cleaned_data['inference']
            overwrite_repetitious = form.cleaned_data['overwrite_repetitious']
            save_concated = True if request.POST.get('save_concated') == "True" else False
            confidence_threshold = form.cleaned_data['confidence_threshold']
            form = SentinelFetchForm(initial={"x_min": x_min, "x_max": x_max, "y_min": y_min,
                                                "y_max": y_max,"zoom": zoom, "lon_min": lon_min,
                                                "lon_max": lon_max, "lat_min": lat_min,
                                                "lat_max": lat_max, "start_date": start_date, "end_date": end_date,
                                                "base_date": base_date, "n_days_before_base_date": n_days_before_base_date,
                                                "overwrite_repetitious": overwrite_repetitious, "preset_area": preset_area,
                                                'inference': inference, "confidence_threshold": confidence_threshold,
                                                })
            return render(request, "fetch_data/SentinelFetch.html", {'form': form, 'user':request.user})
        logging.info("\n\n\n\Last 10 days form is not valid\n\n\n")


def test(request):
    if request.method == 'GET':
        form = SentinelFetchForm(initial={"x_min": 21390, "x_max": 21400, "y_min": 14030, "y_max": 14035, "zoom": 15, "image_store_path": r"D:\SatteliteImages_db"})
        return render(request, "fetch_data/test.html", context={'preset_araes': PresetArea.objects.all(),'form': form,'user':request.user})
    


class territory_fetch_APIView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        territory_serializer = TerritorySerializer(data=request.data)
        print(type(territory_serializer))
        print(territory_serializer)
        print("111")
        print("request.data['preset_area']:", request.data['preset_area'])
        if territory_serializer.is_valid():
            if request.data['preset_area'] != None:
                preset_area_obj = PresetArea.objects.get(tag=request.data['preset_area'])
                # preset_area_obj = request.data['preset_area']
                x_min, x_max, y_min, y_max, zoom = preset_area_obj.x_min, preset_area_obj.x_max, preset_area_obj.y_min, preset_area_obj.y_max, preset_area_obj.zoom
            else:
                x_min = request.data.get('x_min', None)
                x_max = request.data.get('x_max', None)
                y_min = request.data.get('y_min', None)
                y_max = request.data.get('y_max', None)
                zoom = int(request.data.get('zoom', None))
            start_date = request.data.get('start_date', None)
            end_date = request.data.get('end_date', None)
            n_days_before_base_date = request.data.get('n_days_before_base_date', None)
            base_date = request.data.get('base_date', None)
            overwrite_repetitious = request.data.get('overwrite_repetitious', None)
            overwrite_repetitious = True if overwrite_repetitious in ["True", "true", "1", True] else False
            x_range = [int(x_min), int(x_max)]
            y_range = [int(y_min), int(y_max)]

            time_interpreted = start_end_time_interpreter(start=start_date, end=end_date, n_days_before_base_date=n_days_before_base_date, base_date=base_date,
                                                        return_formatted_only=False)
            start_date, end_date = time_interpreted[0][0], time_interpreted[1][0]

            territory_fetch_inference(x_range, y_range, zoom, start=start_date, end=end_date, overwrite_repetitious=overwrite_repetitious, )

            request_parameters = {"x_min": x_min, "y_min": y_min, "y_max": y_max, "zoom": zoom, "start_date": start_date, "end_date": end_date,
                                "overwrite_repetitious": overwrite_repetitious}

            return Response(data={"message": "Images stored successfully",
                         "request_parameters": request_parameters
                         }, status=200)
        error_messages = territory_serializer.errors
        return Response(data={"message": error_messages}, status=400)
    

@login_required(login_url='users:login')
def ConvertView(request):
    if request.method == 'GET':
        logging.info("GET 1")
        form_2xy = Convert2xyForm()
        form_2lonlat = Convert2lonlatForm()
        return render(request, "fetch_data/Conversions.html", context={'form_2xy': form_2xy,
        "form_2lonlat": form_2lonlat,'user':request.user})

    elif request.method == 'POST' and 'convert_2lonlat' in request.POST and request.user.is_authenticated:
        form_2lonlat = Convert2lonlatForm(request.POST)
        form_2xy = Convert2xyForm(request.POST)
        decimal_round = int(request.POST.get('decimal_round')) if request.POST.get('decimal_round')!=None else 6
        if form_2lonlat.is_valid():
            zoom_2lonlat = form_2lonlat.cleaned_data['zoom_2lonlat']
            x_min_2lonlat = form_2lonlat.cleaned_data['x_min_2lonlat']
            x_max_2lonlat = form_2lonlat.cleaned_data['x_max_2lonlat']
            y_min_2lonlat = form_2lonlat.cleaned_data['y_min_2lonlat']
            y_max_2lonlat = form_2lonlat.cleaned_data['y_max_2lonlat']

            x_range_2lonlat = (x_min_2lonlat, x_max_2lonlat)
            y_range_2lonlat = (y_min_2lonlat, y_max_2lonlat)
            lon_min_2lonlat, lat_min_2lonlat, lon_max_2lonlat, lat_max_2lonlat = xyz2bbox_territory(x_range_2lonlat, y_range_2lonlat, zoom_2lonlat)
            lon_min_2lonlat, lat_min_2lonlat, lon_max_2lonlat, lat_max_2lonlat = list(map(lambda x: round(x, decimal_round), (lon_min_2lonlat, lat_min_2lonlat, lon_max_2lonlat, lat_max_2lonlat)))
            convert_2lonlat_res = {'lon_min_2lonlat': lon_min_2lonlat, 'lat_min_2lonlat': lat_min_2lonlat, 'lon_max_2lonlat': lon_max_2lonlat, 'lat_max_2lonlat': lat_max_2lonlat}

            return render(request, "fetch_data/Conversions.html", context={'form_2xy': form_2xy,
            "form_2lonlat": form_2lonlat, 'convert_2lonlat_res': convert_2lonlat_res ,'user':request.user})


    elif request.method == 'POST' and 'convert_2xy' in request.POST and request.user.is_authenticated:
        logging.info("before form_2xy validation")
        print("before form_2xy validation")
        form_2xy = Convert2xyForm(request.POST)
        form_2lonlat = Convert2lonlatForm(request.POST)
        logging.info("before form_2xy validation")
        if form_2xy.is_valid():
            zoom_2xy = form_2xy.cleaned_data['zoom_2xy']
            lon_min_2xy = form_2xy.cleaned_data['lon_min_2xy']
            lon_max_2xy = form_2xy.cleaned_data['lon_max_2xy']
            lat_min_2xy = form_2xy.cleaned_data['lat_min_2xy']
            lat_max_2xy = form_2xy.cleaned_data['lat_max_2xy']

            coords_2xy = (lon_min_2xy, lat_min_2xy, lon_max_2xy, lat_max_2xy)
            (x_min_2xy, x_max_2xy), (y_min_2xy, y_max_2xy), zoom_2xy = coords_2_xyz_newton(coords_2xy, zoom_2xy)
            convert_2xy_res = {"x_min_2xy": x_min_2xy, "x_max_2xy": x_max_2xy, 'y_min_2xy': y_min_2xy, 'y_max_2xy': y_max_2xy, 'zoom_2xy': zoom_2xy}
            print("convert_2xy_res:", convert_2xy_res)

            return render(request, "fetch_data/Conversions.html", context={'form_2xy': form_2xy,
            "form_2lonlat": form_2lonlat, 'convert_2xy_res': convert_2xy_res, 'user':request.user})
    

@login_required(login_url='users:login')
def TasksTable(request, mode, time_limit):
    user = request.user
    if request.method=="GET" and request.user.is_authenticated:
        time_limit = datetime.datetime.today() - timedelta(days=time_limit)
        if mode == 'my':
            all_parent_tasks =  QueuedTask.objects.filter(user_queued=user, is_parent=True, time_queued__gt=time_limit).order_by('-time_queued')
            all_tasks = False
        elif mode == 'all':
            all_parent_tasks =  QueuedTask.objects.filter(is_parent=True, time_queued__gt=time_limit).order_by('-time_queued')
            all_tasks = True
        paginator = Paginator(all_parent_tasks, 5)
        num_pages = paginator.num_pages
        page_number = request.GET.get('page')
        try:
            page_parent_tasks = paginator.get_page(page_number)  # returns the desired page object
        except PageNotAnInteger:   # if page_number is not an integer then assign the first page
            page_parent_tasks = paginator.page(1)
        except EmptyPage:    # if page is empty then return last page
            page_parent_tasks = paginator.page(paginator.num_pages)

        return render(request, "fetch_data/Fetches_table.html", context={'page_parent_tasks': page_parent_tasks, 'user':user, 'all_tasks': all_tasks,
                                                                         "num_pages": num_pages, "pages_range": paginator.page_range,})


@login_required(login_url='users:login')
def TaskResult(request, task_id, filters):
    user = request.user
    filters_dict = dict()

    # extract filter paramters and their values
    pattern = r"L\[min\]=(\d+)&L\[max\]=(\d+)"
    matches = re.search(pattern, filters)
    if matches:
        l_min = matches.group(1)
        l_max = matches.group(2)
        l_min = 0 if l_min in [None, "", " "] else l_min
        l_max = 620 if l_max in [None, "", " "] else l_max
        l_min, l_max = int(l_min), int(l_max)
        filters_dict["L"] = (l_min, l_max)

    if request.method=="GET" and request.user.is_authenticated:
        task = QueuedTask.objects.get(task_id=task_id)
        if "L" in filters:
            l_min, l_max = filters_dict["L"]
            detected_objects = task.detected_objects.filter(length__gte=l_min, length__lte=l_max).order_by('-length')
        else:
            detected_objects = task.detected_objects.all().order_by('-length')
        n_objects = len(detected_objects)
        paginator = Paginator(detected_objects, 25)
        num_pages = paginator.num_pages
        page_number = request.GET.get('page')
        try:
            page_objects = paginator.get_page(page_number)  # returns the desired page object
        except PageNotAnInteger:   # if page_number is not an integer then assign the first page
            page_objects = paginator.page(1)
        except EmptyPage:    # if page is empty then return last page
            page_objects = paginator.page(paginator.num_pages)
        context={'task': task, 'page_objects': page_objects, "n_objects": n_objects, "num_pages": num_pages, "pages_range": paginator.page_range,  }
        if task.parent_task.all().first() != None:
            parent_task = task.parent_task.all().first().task_id
            context["parent_task"] = parent_task
        if "L" in filters:
            context["L"] = (l_min, l_max)
        return render(request, "fetch_data/Task_Result.html", context=context)


@login_required(login_url='users:login')
def ImageGet(request, task_id, image_dir):
    image_dir_annotated = image_dir[:-4] + "_annotated" + image_dir[-4:]
    img_attrs = image_dir_annotated.split("\\")
    x_str, y_str = img_attrs[-3], img_attrs[-2]
    image_name = f"X{x_str}_Y{y_str}_{os.path.basename(image_dir_annotated)}"
    file = open(image_dir_annotated, 'rb')
    response = HttpResponse(file, content_type="image/png")
    response["Content-Disposition"] = f"attachment; filename={image_name}"
    return response


@login_required(login_url='users:login')
def ConcatImage(request, mode, task_id):
    from fetch_data.utilities.image_db import images_db_path
    if mode in ['normal', 'annot']:
        annotated = mode == 'annot'
    task = QueuedTask.objects.get(task_id=task_id) 
    x_range, y_range, zoom = (task.x_min, task.x_max), (task.y_min , task.y_max), task.zoom
    if (x_range[1] - x_range[0] + 1) * (y_range[1] - y_range[0] + 1) > concat_size_limit:
        messages.warning(request, "Area is too large for image concatenation!")
        return HttpResponseRedirect(reverse('fetch_data:task_result' , kwargs={"task_id": task_id, "filters":"None"}))
    start, end = task.time_from, task.time_to

    from fetch_data.utilities.image_db import concated_images_path
    if not os.path.exists(concated_images_path):
        os.mkdir(concated_images_path)
    concated_img_path = concatenate_image(x_range, y_range, zoom, start, end, annotated=annotated, images_db_path=images_db_path, return_img=False,
                    save_img=True, save_img_path=concated_images_path)
    file = open(concated_img_path, 'rb')
    annotated_suffix = "_annotated" if mode == "annot" else ""
    concated_img_name = f"concated_X[{x_range[0]}-{x_range[1]}]_Y[{y_range[0]}-{y_range[1]}]{annotated_suffix}.png"
    response = HttpResponse(file, content_type="image/png")
    response["Content-Disposition"] = f"attachment; filename={concated_img_name}"
    return response



@login_required(login_url='users:login')
def CustomAnnotation(request, task_id):
    from fetch_data.utilities.imageutils import draw_bbox_torchvision
    from fetch_data.utilities.image_db import images_db_path
    l_min = request.POST.get('l_min')
    l_max = request.POST.get('l_max')
    l_min = 0 if l_min in [None, "", " "] else l_min
    l_max = 620 if l_max in [None, "", " "] else l_max
    l_min, l_max = int(l_min), int(l_max)
    print("\n\n\nL Min: ", l_min)
    print("\n\n\nL Max: ", l_max)
    task = QueuedTask.objects.get(task_id=task_id)
    x_range, y_range, zoom = (task.x_min, task.x_max), (task.y_min , task.y_max), task.zoom
    concated_size = (x_range[1] - x_range[0] + 1) * (y_range[1] - y_range[0] + 1)
    if (len(task.child_task.all()) == 0) or (concated_size < concat_size_limit):  # in case task is a child or is a parent with relative normal size (not excessive large size)
        start, end = task.time_from, task.time_to
        concated_img = concatenate_image(x_range, y_range, zoom, start, end, annotated=False, images_db_path=images_db_path, return_img=True,
                        save_img=False,)
        inference_result = task.inference_result
        inference_result = json.loads(inference_result)
        bboxes = np.array(inference_result["bboxes"])
        scores = np.array(inference_result["scores"])
        ships_coords = inference_result["coords"]
        lengths = inference_result["lengths"]
        constraints = dict()
        constraints["length"] = (l_min, l_max)
        from fetch_data.utilities.image_db import concated_images_path
        if not os.path.exists(concated_images_path):
            os.mkdir(concated_images_path)
        custom_annotated_img_name = f"concated_X[{x_range[0]}-{x_range[1]}]_Y[{y_range[0]}-{y_range[1]}]_custom_annotated.png"
        custom_annotated_img_path = os.path.join( concated_images_path ,custom_annotated_img_name)
        draw_bbox_torchvision(concated_img, bboxes, scores, lengths=lengths, ships_coords=ships_coords, annotations=["score", "length", "coord"], save=True,
                          image_save_name=custom_annotated_img_path, output_annotated_image=False, font_size=14, font=r"calibri.ttf", bbox_width=2, constraints=constraints)
        
        file = open(custom_annotated_img_path, 'rb')
        response = HttpResponse(file, content_type="image/png")
        response["Content-Disposition"] = f"attachment; filename={custom_annotated_img_name}"
        return response
    else:
        messages.warning(request, f"Area is too large for custom inferencing! This area consists of {concated_size} tiles while there is a limitation of {concat_size_limit} tiles to be concatenated on server.")
        return HttpResponseRedirect(reverse('fetch_data:task_result', kwargs={"task_id": task_id, "filters":"None" }))
