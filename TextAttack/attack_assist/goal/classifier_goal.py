from .base import AttackGoal

class ClassifierGoal(AttackGoal):
    def __init__(self, target, targeted):
        self.target = target
        self.targeted = targeted
    
    @property
    def is_targeted(self):
        return self.targeted

    def check(self, adversarial_sample, prediction):
        if self.targeted:
            return prediction == self.target
        else:
            print("class_goal-------",adversarial_sample,self.target,prediction)
            return prediction != self.target