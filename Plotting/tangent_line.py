def tangent_line(f,x_0,x_start,x_end,linestyle):
  epsilon = 0.0000001
  deriv = ((f(x_0+epsilon) - f(x_0)) / epsilon)
  x = np.linspace(x_start,x_end,200)
  y = f(x) 
  y_0 = f(x_0)
  y_tan = deriv * (x - x_0) + y_0 
  plt.plot(x,y_tan,linestyle)