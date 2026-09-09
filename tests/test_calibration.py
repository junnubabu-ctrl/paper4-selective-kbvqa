import numpy as np
from paper4_kbvqa.calibration.calibrators import PlattCalibrator, TemperatureCalibrator

def test_platt_output_bounds():
    p=PlattCalibrator().fit([.1,.2,.8,.9],[0,0,1,1]).predict([.3,.7])
    assert np.all((p>=0)&(p<=1))
def test_temperature_positive():
    c=TemperatureCalibrator().fit([-2,-1,1,2],[0,0,1,1]); assert c.temperature>0
