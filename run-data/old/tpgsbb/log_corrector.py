import pandas as pd

outfile = "log.csv"

df_tpg = pd.read_csv("log-tpg.csv").rename(columns=lambda x: x.strip())
df_sbb = pd.read_csv("log-sbb.csv").rename(columns=lambda x: x.strip())
df_out = pd.DataFrame([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]], columns=df_tpg.columns)

sbb_gens = df_sbb["gen"].unique().tolist()

cur_tpg_gen = 0

# construct log line by line
while True:

    # SBB segment
    if len(sbb_gens) > 0 and cur_tpg_gen in sbb_gens:
    
        cur_sbb_gens = df_sbb[df_sbb["gen"] == sbb_gens[0]]["gen_b"].max()
        sbb_gens.pop(0)
        
        # copy last TPG gen line for current SBB run
        copy_line = df_out.iloc[-1]
        copy_line.iloc[[1, -1, -2, -5, -6]] = 0
        for i in range(cur_sbb_gens):
            copy_line.iloc[0] = copy_line.iloc[0] + 1
            df_out = df_out.append(copy_line)

    else:
        # regular TPG segment
        gen = df_out.iloc[-1, 0] + 1
        tpg_line = df_tpg.iloc[cur_tpg_gen]
        cur_tpg_gen += 1
        tpg_line.iloc[0] = gen
        df_out = df_out.append(tpg_line)
        
    if cur_tpg_gen >= len(df_tpg):
        break
        
df_out.to_csv(outfile, index=False)
