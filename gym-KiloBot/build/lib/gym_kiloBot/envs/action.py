
class Action:
    def __init__(self,theta=0,r=0):
        self.theta = theta
        self.r = r
    def norm(self):
        return self.radius
        ## This should actually give out the actuall length travelled
        ## based on physics of the particle for now this is fine
