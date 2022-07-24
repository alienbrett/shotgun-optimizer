
import numpy as np
from dataclasses import dataclass
import typing



class VectorSpaceTranslator:
    def __init__(self, ignore_mask : np.ndarray):
        '''Mask vec should be 1 where space should be truncated
        '''
        self._n = len(ignore_mask)
        self._k = int(np.sum(1-np.abs(ignore_mask)))
        self._mask = ignore_mask
        self._index = np.where(1-ignore_mask)[0]
    

    def outer_size(self,) -> int:
        return self._n
    
    def inner_size(self,) -> int:
        return self._k

    
    def _gen_idx_broadcast_shape(self, n_dims, axis, k=None) -> typing.List[int]:
        if k is None:
            k = self._k
        shape = [ (k if x == axis else 1)  for x in range(n_dims)]
        return shape
    
    
    
    def encode(self, vec_n: np.ndarray, axis:int =0) -> np.ndarray:
        vec_n = np.asarray(vec_n)
        n = len(vec_n.shape)
        
        idx = self._index.reshape(
            tuple(self._gen_idx_broadcast_shape(n, axis, k=None))
        )
        
        res_arr = np.take_along_axis(vec_n, idx, axis=axis)
        return res_arr
    
    
    
    def decode(self, vec_m: np.ndarray, axis:int =0) -> np.ndarray:
        vec_m = np.asarray(vec_m)
        n = len(vec_m.shape)
        
        new_s = list(vec_m.shape)
        new_s[axis] = self._n
        new_s = tuple(new_s)

        res_array = np.zeros(
            # self._gen_idx_broadcast_shape(n, axis, k=self._n)
            new_s,
        )
        idx = self._index.reshape(
            self._gen_idx_broadcast_shape(n,axis,k=None)
        )
        # print('res arr:',res_array.shape)
        # print('idx:',idx.shape)
        # print('vec_m:',vec_m.shape)
        # print('axis:',axis)
        np.put_along_axis(
            res_array,
            idx,
            vec_m,
            axis
        )
        return res_array