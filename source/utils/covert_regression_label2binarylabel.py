import pandas as pd

data = pd.read_csv("/local/home/Anonymous/AURORA/AURORA_F4_yang_clinicaloutx.csv")


# PRE_PROM_Dep8b_T_i
# PRE_PROM_AnxBank_RS_i
# M6_PROM_Dep8b_T_i
# M6_PROM_AnxBank_RS_i

dep_thres = 0

anx_thres = 0

dep_label =  data["M6_PROM_Dep8b_T_i"] - data["PRE_PROM_Dep8b_T_i"]

dep_label = dep_label.apply(lambda x: 1 if x > dep_thres else 0)

# print distribution
print('dep_new', dep_label.value_counts())

data['dep_new'] = dep_label

anx_label =  data["M6_PROM_AnxBank_RS_i"] - data["PRE_PROM_AnxBank_RS_i"]
anx_label = anx_label.apply(lambda x: 1 if x > anx_thres else 0)

print('anx_new', anx_label.value_counts())

data['anx_new'] = anx_label

data.to_csv("/local/home/Anonymous/AURORA/AURORA_F4_yang_clinicaloutx_new.csv", index=False)