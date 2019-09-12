import pandas as pd

# function to extract 5 column set, analyze for duplicates, and recode for 1-5 scale
def processQuestion(master_df, agree_lot, agree_little, neither, dis_little, dis_lot, caption='Question X'):
    df_temp = master_df[[agree_lot, agree_little, neither, dis_little, dis_lot]]
    df_temp['total'] = df_temp.sum(axis=1)  # total column
    print(f'\nFrequency of {caption} columns:\n {df_temp[df_temp == 1].count()}')
    print(f'\nNumber of check boxes marked for {caption} columns:\n {df_temp["total"].value_counts()}')
    df_temp[agree_lot].replace(1,5, inplace=True)
    df_temp[agree_little].replace(1, 4, inplace=True)
    df_temp[neither].replace(1, 3, inplace=True)
    df_temp[dis_little].replace(1, 2, inplace=True)
    df_temp['scale_value'] = df_temp[[agree_lot, agree_little, neither, dis_little, dis_lot]].apply(max, axis=1)
    print(f'\nFrequency of {caption}  after 1-5 scale:\n {df_temp["scale_value"].value_counts()}')
    return df_temp

# import file and rename columns for I'M 1ST OF FRNDS HAVE NEW ELCTRNC EQUIP
# and CMPTRS CONFUSE ME,NEVER GET USED TO THEM
colspecs = [[0, 7], [6945, 6946], [6962, 6963], [6996, 6997], [7013, 7014], [7030, 7031],
            [6950, 6951], [6967, 6968], [7001, 7002], [7018, 7019], [7035, 7036]]
df = pd.read_fwf('FA15_Data.txt', colspecs=colspecs, encoding='utf8')
df.columns = ['id', 'ftech_agree_lot', 'ftech_agree_little', 'ftech_neither', 'ftech_dis_little', 'ftech_dis_lot',
              'compconf_agree_lot', 'compconf_agree_little', 'compconf_neither', 'compconf_dis_litte', 'compconf_dis_lot']

# fill in missing values with 0
df.fillna(0, inplace=True)


# convert all the columns to int
convert_dict = {'id':int, 'ftech_agree_lot':int, 'ftech_agree_little':int,
                'ftech_neither':int, 'ftech_dis_little':int, 'ftech_dis_lot':int,
                'compconf_agree_lot':int, 'compconf_agree_little':int,
                'compconf_neither':int, 'compconf_dis_litte':int, 'compconf_dis_lot':int}
df = df.astype(convert_dict)


# process I'M 1ST OF FRNDS HAVE NEW ELCTRNC EQUIP
df_tech = processQuestion(df,'ftech_agree_lot', 'ftech_agree_little', 'ftech_neither',
                          'ftech_dis_little', 'ftech_dis_lot', 'First In Tech')

# process CMPTRS CONFUSE ME,NEVER GET USED TO THEM
df_conf = processQuestion(df,'compconf_agree_lot', 'compconf_agree_little', 'compconf_neither',
                          'compconf_dis_litte', 'compconf_dis_lot', 'Computers Confuse')

