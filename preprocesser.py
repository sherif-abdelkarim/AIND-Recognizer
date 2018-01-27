import numpy as np
import pandas as pd
from asl_data import AslDb
import math

asl = AslDb() # initializes the database
asl.df['grnd-ry'] = asl.df['right-y'] - asl.df['nose-y']
# TODO add df columns for 'grnd-rx', 'grnd-ly', 'grnd-lx' representing differences between hand and nose locations
asl.df['grnd-rx'] = asl.df['right-x'] - asl.df['nose-x']
asl.df['grnd-ly'] = asl.df['left-y'] - asl.df['nose-y']
asl.df['grnd-lx'] = asl.df['left-x'] - asl.df['nose-x']
features_ground = ['grnd-rx','grnd-ry','grnd-lx','grnd-ly']
df_means = asl.df.groupby('speaker').mean()
asl.df['left-x-mean']= asl.df['speaker'].map(df_means['left-x'])
df_std = asl.df.groupby('speaker').std()
features_norm = ['norm-rx', 'norm-ry', 'norm-lx','norm-ly']
asl.df['norm-rx'] = (asl.df['right-x'] - asl.df['speaker'].map(df_means['right-x']))/asl.df['speaker'].map(df_std['right-x'])
asl.df['norm-ry'] = (asl.df['right-y'] - asl.df['speaker'].map(df_means['right-y']))/asl.df['speaker'].map(df_std['right-y'])
asl.df['norm-lx'] = (asl.df['left-x'] - asl.df['speaker'].map(df_means['left-x']))/asl.df['speaker'].map(df_std['left-x'])
asl.df['norm-ly'] = (asl.df['left-y'] - asl.df['speaker'].map(df_means['left-y']))/asl.df['speaker'].map(df_std['left-y'])
features_polar = ['polar-rr', 'polar-rtheta', 'polar-lr', 'polar-ltheta']
polar_r = lambda x,y: np.sqrt(x**2 + y**2)
polar_theta = lambda x,y: np.arctan2(x,y)
asl.df['polar-rr'] = polar_r(asl.df['grnd-rx'], asl.df['grnd-ry'])
asl.df['polar-rtheta'] = polar_theta(asl.df['grnd-rx'], asl.df['grnd-ry'])
asl.df['polar-lr'] = polar_r(asl.df['grnd-lx'], asl.df['grnd-ly'])
asl.df['polar-ltheta'] = polar_theta(asl.df['grnd-lx'], asl.df['grnd-ly'])

# TODO add features for left, right, x, y differences by one time step, i.e. the "delta" values discussed in the lecture
# Name these 'delta-rx', 'delta-ry', 'delta-lx', and 'delta-ly'

features_delta = ['delta-rx', 'delta-ry', 'delta-lx', 'delta-ly']
asl.df['delta-rx'] = [n for n in asl.df.groupby('speaker', sort=False)['right-x'].diff()]
asl.df['delta-ry'] = [n for n in asl.df.groupby('speaker', sort=False)['right-y'].diff()]
asl.df['delta-lx'] = [n for n in asl.df.groupby('speaker', sort=False)['left-x'].diff()]
asl.df['delta-ly'] = [n for n in asl.df.groupby('speaker', sort=False)['left-x'].diff()]
asl.df = asl.df.fillna(0)

# TODO add features of your own design, which may be a combination of the above or something else
# Name these whatever you would like

# TODO define a list named 'features_custom' for building the training set
features_custom = ['pnorm-rr', 'pnorm-rtheta', 'pnorm-lr','pnorm-ltheta']
asl.df['pnorm-rr'] = polar_r(asl.df['norm-rx'], asl.df['norm-ry'])
asl.df['pnorm-rtheta'] = polar_theta(asl.df['norm-rx'], asl.df['norm-ry'])
asl.df['pnorm-lr'] = polar_r(asl.df['norm-lx'], asl.df['norm-ly'])
asl.df['pnorm-ltheta'] = polar_theta(asl.df['norm-lx'], asl.df['norm-ly'])
