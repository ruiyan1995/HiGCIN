import torch
import numpy as np

class Mask(object):
	"""docstring for Mask"""
	def __init__(self, mask_type, T, S):
		super(Mask, self).__init__()
		self.T = T
		self.S = S
		self.mask_type = mask_type
		self.adj = 1
		
	
	def get_personal_mask(self, t, s):
		mask = torch.zeros(self.T, self.S)
		if self.mask_type == 'Spatial':
			mask[t,:] = 1
		elif self.mask_type == 'Cross':
			mask[t,:] = 1
			mask[:,s] = 1
		elif self.mask_type == 'All':
			mask[:,:] = 1
		elif self.mask_type == 'Adjacent_frame':
			adj = self.adj
			interval = 2*adj+1
			if t-adj<0:
				start = 0
			elif t+adj+1>self.T:
				start = self.T - interval
			else:
				start = t-adj
			mask[start:start+interval,:] = 1
		elif self.mask_type == 'Gradient':
			pass
		return mask

	def get_RelationMap(self):
		RelationMap = torch.zeros(self.T*self.S, self.T*self.S)
		for t in range(self.T):
			for s in range(self.S):
				mask = self.get_personal_mask(t, s)
				#print t
				#show_map(mask)
				self.set_map(t, s, RelationMap, mask)
		return RelationMap


	def set_map(self, t, s, RelationMap, mask):
		assert mask is not None, 'mask is None!'
		assert mask.size() == (self.T, self.S), 'dim should be (num_players, num_frames)'
		for _t in range(mask.size(0)):
			for _s in range(mask.size(1)):
				if mask[_t][_s]:
					#print t,s,_t,_s
					RelationMap[t*self.S+s][_t*self.S+_s] = 1
					#RelationMap[_s*_t][s*t] = 1

def show_map(Map):
	for line in Map:
		str_line = ''
		for e in line:
			if int(e) == 1:
				str_line = str_line + '\033[1;35m'+str(int(e))+'\033[0m'
			else:
				str_line = str_line + '\033[1;36m'+str(int(e))+'\033[0m'
		print (str_line)

if __name__ == '__main__':
	np.set_printoptions(threshold='nan', linewidth=1000)
	RelationMap = Mask('Spatial').get_RelationMap()
	Map = RelationMap.numpy().tolist()
	show_map(Map)
	