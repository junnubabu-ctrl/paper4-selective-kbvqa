from __future__ import annotations
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression

class PlattCalibrator:
    def __init__(self): self.model=LogisticRegression(solver="lbfgs")
    def fit(self, scores, correct):
        x=np.asarray(scores,dtype=float).reshape(-1,1); y=np.asarray(correct,dtype=int)
        if len(np.unique(y))<2: raise ValueError("Calibration labels require both correct and incorrect examples")
        self.model.fit(x,y); return self
    def predict(self,scores): return self.model.predict_proba(np.asarray(scores,dtype=float).reshape(-1,1))[:,1]
    def to_dict(self):
        if not hasattr(self.model, "coef_"):
            raise RuntimeError("Calibrator must be fitted before serialization")
        return {"type":"platt","coef":float(self.model.coef_[0,0]),"intercept":float(self.model.intercept_[0])}
    @classmethod
    def from_dict(cls, data):
        if data.get("type") != "platt":
            raise ValueError("Not a Platt calibrator payload")
        obj=cls()
        obj.model.classes_=np.asarray([0,1],dtype=int)
        obj.model.coef_=np.asarray([[float(data["coef"])] ]); obj.model.intercept_=np.asarray([float(data["intercept"])])
        obj.model.n_features_in_=1
        return obj

class IsotonicCalibrator:
    def __init__(self): self.model=IsotonicRegression(out_of_bounds="clip")
    def fit(self,scores,correct): self.model.fit(np.asarray(scores,float),np.asarray(correct,int)); return self
    def predict(self,scores): return np.asarray(self.model.predict(np.asarray(scores,float)),float)

class TemperatureCalibrator:
    """Temperature scaling for binary logits; grid-searches T on validation NLL."""
    def __init__(self): self.temperature=1.0
    @staticmethod
    def _sigmoid(x): return 1/(1+np.exp(-np.clip(x,-50,50)))
    def fit(self,logits,correct):
        z=np.asarray(logits,float); y=np.asarray(correct,float)
        candidates=np.logspace(-2,2,401); best=(float('inf'),1.0)
        for t in candidates:
            p=np.clip(self._sigmoid(z/t),1e-7,1-1e-7)
            nll=float(-(y*np.log(p)+(1-y)*np.log(1-p)).mean())
            if nll<best[0]: best=(nll,float(t))
        self.temperature=best[1]; return self
    def predict(self,logits): return self._sigmoid(np.asarray(logits,float)/self.temperature)
