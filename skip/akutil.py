import numpy as np

class AKLog:
    def __init__(self, ):
        self.ts2data = {}
        self.ts = 0
        
    def put(self, ts=None, use_np=True, **kwargs):
        """
        Add data given a timestep, a key, and a value.
        """
        if ts is None:
            ts = self.ts
        assert isinstance(ts, int)
        if ts not in self.ts2data:
            self.ts2data[ts] = {}
        
        if use_np:
            kwargs = {key: np.asarray(value) for key, value in kwargs.items()}
        self.ts2data[ts].update(kwargs)
        
    def next_ts(self):
        self.ts += 1
        
    def keys(self):
        return set.union(*[set(data.keys()) for data in self.ts2data.values()])
    
    def get_array(self, key, ret_ts=True, use_np_stack=True):
        """
        return the ts and data array for a given key
        """
        if ret_ts:
            ts = np.stack([ts for ts, tsdata in self.ts2data.items() if key in tsdata])
        y = [tsdata[key] for ts, tsdata in self.ts2data.items() if key in tsdata]
        if use_np_stack:
            y = np.stack(y, axis=0)
        return (ts, y) if ret_ts else y
        
    def get_arrays(self, *keys):
        """
        return a dictionary for the ts and data arrays.
        """
        data = {}
        if len(keys)==0:
            # all keys
            keys = set.union(*[set(data.keys()) for data in self.ts2data.values()])
        data['ts'] = np.stack([list(self.ts2data.keys())])
        for key in keys:
            a = [tsdata[key] if key in tsdata else None for ts, tsdata in self.ts2data.items()]
            try:
                a = np.stack(a)
            except:
                pass
            data[key] = a
        return data
        