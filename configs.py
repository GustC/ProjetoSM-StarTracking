import cv2 as cv
feature_params_day = dict( maxCorners = 50,
                       qualityLevel = 0.1,
                       minDistance = 30,
                       blockSize = 50 )
lk_params_day = dict( winSize  = (10,10),
                  maxLevel = 10,
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 30, 0.01))
feature_params_night = dict( maxCorners = 50,
                       qualityLevel = 0.2,
                       minDistance = 40,
                       blockSize = 100 )
lk_params_night = dict( winSize  = (60,60),
                  maxLevel = 10,
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 30, 0.01))

typeDayNight = "night"
typeDayNight = "day"
class ConfigApp:
    def __init__(self,typeDay,qtyStars):
        if(typeDay == typeDayNight):
            if(qtyStars > 100):
                qtyStars = 100
            feature_params_night["maxCorners"] = qtyStars
            self.configFeature = feature_params_night
            self.configLk = lk_params_night
        else:
            if(qtyStars > 100):
                qtyStars = 100
            feature_params_day["maxCorners"] = qtyStars
            self.configFeature = feature_params_day
            self.configLk = lk_params_day