### Search for specific phrases in an object methods

def methods_lookup(search_strings, object):
    my_list = dir(object)
    matching_strings = []

    for search_str in search_strings:
        derivs = [search_str.capitalize(), search_str.lower(), search_str.upper()]
        for deriv in derivs:
            matched = [s for s in my_list if deriv in s]
            matching_strings = matching_strings + matched

    return matching_strings


""" Example

search_strings = ["right", "left", "rtl", "ltr", "format", "Paragraph", "direction", "dir", "align"]
object = paragraph.Format

methods_lookup(search_strings, object)

>>> ['RightIndent',
 'SetRightIndent',
 'LeftIndent',
 'SetLeftIndent',
 'ClearFormatting',
 '__format__',
 '__dir__',
 'HorizontalAlignment',
 'TextAlignment']

"""