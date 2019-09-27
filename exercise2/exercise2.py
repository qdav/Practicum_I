import pandas as pd
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_kmo
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np




class Question:
    def __init__(self, question_txt, question_column, col_range):
        self.question_txt = question_txt
        self.question_column = question_column
        self.col_range = col_range

        self.df_question = pd.DataFrame

    def get_column_names(self):
        pass

    def get_column_range(self):
        return self.col_range

    def load_question(self, master_df):
        self.df_question = master_df[self.get_column_names()]

    def get_question_fequency(self):
        df_temp = self.df_question.copy()
        df_temp['total'] = df_temp.sum(axis=1)  # total column
        print(f'\nFrequency of {self.question_column} columns:\n {df_temp[df_temp == 1].count()}')

    def count_answers(self):
        df_temp = self.df_question.copy()
        df_temp['total'] = df_temp.sum(axis=1)  # total column
        print(f'\nNumber of check boxes marked for {self.question_column} columns:\n {df_temp["total"].value_counts()}')

    def get_final_value(self):
        pass

    def evaluate_question(self):
        """Perform validation and summarization of the question."""
        self.get_question_fequency()
        self.count_answers()


class CategoryQuestion(Question):
    def __init__(self, question_txt, question_prefix, col_range, col_value, use_strings=False):
        """Create a Question object by passing in the location of the columns and other metadata."""
        super().__init__(question_txt, question_prefix, col_range)
        self.col_value = col_value
        self.use_strings = use_strings

    def get_column_names(self):
        # here, creating combined column/volue column names for uniqueness
        colname_temp = list()
        for column in self.col_value:
            colname_temp.append(self.question_column + "-" + str(column))
        return colname_temp


    def get_final_value(self):
        df_temp = self.df_question.copy()

        # TODO: enhance this logic to handle duplicate values better
        # currently, it picks the highest alphabetical value across a row
        for column in df_temp:
            df_temp.loc[df_temp[column] == 1, column] = column.replace((self.question_column + '-'), '')

        if self.use_strings:
            df_temp = df_temp.astype(str)
        else:
            df_temp = df_temp.astype(int)

        df_temp[self.question_column] = df_temp.iloc[:, :].apply(max, axis=1)
        print(f'\nFrequency of {self.question_column} after scale:\n {df_temp[self.question_column].value_counts()}')
        return df_temp[self.question_column]

class SingleValueQuestion(Question):
    def get_column_names(self):
        names = []
        names.append(self.question_column)
        return names

    def get_final_value(self):
        df_temp = self.df_question.copy()
        print(f'\nFrequency of {self.question_column}:\n {df_temp[self.question_column].value_counts()}')
        return df_temp[self.question_column]


class OpinionQuestion(Question):
    """ support for 5-answer survey questions from fixed-width file """
    def __init__(self, question_txt, question_prefix, col_range):
        """Create a Question object by passing in the location of the columns and other metadata."""
        super().__init__(question_txt, question_prefix, col_range)
        self.agree_lot_rng = col_range[0]
        self.agree_little_rng = col_range[1]
        self.neither_rng = col_range[2]
        self.dis_little_rng = col_range[3]
        self.dis_lot_rng = col_range[4]

    def get_column_names(self):
        """Helper function to return column names for 5-answer survey."""
        names = []
        names.append(self.question_column + "_agree_lot")
        names.append(self.question_column + "_agree_little")
        names.append(self.question_column + "_neither")
        names.append(self.question_column + "_dis_little")
        names.append(self.question_column + "_dis_lot")
        return names

    def get_final_value(self):
        """Convert five column answers into a single 1-5 value."""
        df_temp = self.df_question.copy()
        df_temp.iloc[:, 0].replace(1, 5, inplace=True)
        df_temp.iloc[:, 1].replace(1, 4, inplace=True)
        df_temp.iloc[:, 2].replace(1, 3, inplace=True)
        df_temp.iloc[:, 3].replace(1, 2, inplace=True)
        df_temp[self.question_column] = df_temp.iloc[:, 0:5].apply(max, axis=1)
        print(f'\nFrequency of {self.question_column} after 1-5 scale:\n {df_temp[self.question_column].value_counts()}')
        return df_temp[self.question_column]

class Survey:
    """Reads in 5-answer servey questions from a fixed-width file and evaluates/processes"""
    def __init__(self, file, question_list, encoding, read_file=True, load_questions=True, verbose=True):
        """File, questions, and encoding for the Survey"""
        self.file = file
        self.question_list = question_list
        self.encoding = encoding

        self.data = pd.DataFrame

        if read_file:
            self.read_file()

        if load_questions:
            self.load_questions(verbose)

    def read_file(self):
        """Reads in survey file based on any specified questions."""
        colspecs = [[0, 7]]  # for the id
        names = ['id']
        for question in self.question_list:
            colspecs.extend(question.get_column_range())
            names.extend(question.get_column_names())

        self.data = pd.read_fwf(self.file, colspecs=colspecs, encoding=self.encoding, names=names, header=None)
        self.data.fillna(0, inplace=True)
        self.data = self.data.astype(int)
        return self.data

    def load_questions(self, verbose=True):
        """Loads any questions of interest in the survey."""
        for question in self.question_list:
            question.load_question(self.data)

    def evaluate_questions(self):
        """Evaluates questions in the survey for frequency, selected answers."""
        for question in self.question_list:
            question.evaluate_question()

    def get_final_values(self):
        """Convert questions into a single 1-5 value."""
        temp_scaled_val = []
        for question in self.question_list:
            test = question.get_final_value()
            temp_scaled_val.append(test)
        return pd.DataFrame(temp_scaled_val).transpose()


class FactorExplore:
    def __init__(self, df, rotation='varimax', method='principal', n_factors=3, impute='drop', verbose=False ):
        """ Build and fit factor model"""
        self.df = df
        self.rotation = rotation
        self.method = method
        self.n_factors = n_factors
        self.impute = impute
        self.verbose = verbose

        # build and fit factor model
        self.fa = FactorAnalyzer(rotation=self.rotation, method=self.method, n_factors=self.n_factors, impute=self.impute)
        self.fa.fit(self.df)
        if self.verbose:
            print(f'\nFactor Model\n {self.fa}')

    def get_barlett_sphericity(self):
        """ Check Bartlett Sphericity """

        chi_square_value, p_value = calculate_bartlett_sphericity(scaled_df)
        if self.verbose:
            print(f'Bartlett Sphericity chi square value: {chi_square_value}\n')
            print(f'Baretlett Sphericity p-value: {p_value}')
        return chi_square_value, p_value

    def get_kmo(self):
        """ Check KMO """
        kmo_all, kmo_model = calculate_kmo(scaled_df)
        if self.verbose:
            print(f'KMO Model:\n{kmo_model}')
        return kmo_all, kmo_model

    def get_factor_loadings(self):
        factor_loadings = pd.DataFrame(self.fa.loadings_)
        factor_loadings.set_index( self.df.columns, inplace=True)
        if self.verbose:
            print(f'Factor Loadings\n{factor_loadings}')
        return factor_loadings

    def get_communalities(self):
        df_communalities = pd.DataFrame(self.fa.get_communalities()).set_index(self.df.columns)
        if self.verbose:
            print(f'Communalities\n{df_communalities}\n')
        return df_communalities

    def get_eigenvalues(self):
        ev, v = self.fa.get_eigenvalues()
        df_eignevalues = pd.DataFrame(ev)
        if self.verbose:
            print(f'Eigenvalues\n{df_eignevalues}\n')
        return df_eignevalues

    def scree_plot(self, ev):
        plt.scatter(range(1,len(ev)+1), ev)
        plt.plot(range(1,len(ev)+1), ev)
        plt.title("Scree Plot")
        plt.xlabel("Factors")
        plt.ylabel("Eigenvalue")
        plt.grid()
        plt.show()

    def get_transformed_data(self, df):
        temp_df = pd.DataFrame(self.fa.transform(df))
        return temp_df

class ClusterExplore():
    def __init__(self ):
        pass

    def plot_elbow(self, df, n_clusters=10):
        # https://towardsdatascience.com/k-means-clustering-algorithm-applications-evaluation-methods-and-drawbacks-aa03e644b48a
        sse = []
        list_k = list(range(1, n_clusters))

        for k in list_k:
            km = KMeans(n_clusters=k)
            km.fit(df)
            sse.append(km.inertia_)
        # Plot sse against k
        plt.figure(figsize=(6, 6))
        plt.plot(list_k, sse, '-o')
        plt.xlabel(r'Number of clusters *k*')
        plt.ylabel('Sum of squared distance')
        plt.show()

    def get_silhouette(self, factor_df, n_clusters=10):
        # https://stackoverflow.com/questions/51138686/how-to-use-silhouette-score-in-k-means-clustering-from-sklearn-library
        range_n_clusters = list(range(2, 10))
        for n_clusters in range_n_clusters:
            clusterer = KMeans(n_clusters=n_clusters)
            preds = clusterer.fit_predict(factor_df)
            #centers = clusterer.cluster_centers_

            score = silhouette_score(factor_df, preds, metric='euclidean')
            print("For n_clusters = {}, silhouette score is {})".format(n_clusters, score))

    def get_cluster_assignments(self, data, n_clusters=10):
        kmeans = KMeans(n_clusters=n_clusters)
        kmeans.fit(data)
        pred = kmeans.predict(data)
        return pd.DataFrame(pred)


# define profile/advertising data
demo_list = list()

#demographcis
demo_list.append(SingleValueQuestion("HOUSEHOLD - $100,000 OR MORE", 'househ100K', [[2690, 2691]]))
demo_list.append(CategoryQuestion("GENDER", 'gender',
                                    [[2382, 2383], [2383, 2384]], [1, 0], use_strings=False))
# where to advertise
demo_list.append(SingleValueQuestion("TRAVEL CHANNEL", 'travchannel', [[9683, 9684]]))
demo_list.append(SingleValueQuestion("NATIONAL GEOGRAPHIC CHANNEL", 'natgeo', [[9655, 9656]]))
demo_list.append(SingleValueQuestion("OUTDOOR CHANNEL", 'outdoorchanel', [[9664, 9665]]))

demo_survey = Survey('FA15_Data.txt', demo_list, 'utf8', verbose=True)
demo_survey.evaluate_questions()
demo_df = demo_survey.get_final_values()


# define factor questions
qlist = list()

# willingness to spend disposable income
qlist.append(OpinionQuestion("BUDGET ALLOWS ME TO BUY DESIGNER CLOTHES", 'designclothes',
                              [[3432, 3433], [3459, 3460], [3513, 3514], [3540, 3541], [3567, 3568]]))
qlist.append(OpinionQuestion("LIKE A NEW CAR EVERY TWO OR THREE YEARS", 'freqnewcar',
                              [[3598, 3599], [3634, 3635], [3706, 3707], [3742, 3743], [3778, 3779]]))
qlist.append(OpinionQuestion("SPEND WHAT I HAVE TO, TO LOOK YOUNGER", 'spendyounger',
                              [[4018, 4019], [4037, 4038], [4075, 4076], [4094, 4095], [4113, 4114]]))
qlist.append(OpinionQuestion("I FEEL FINANCIALLY SECURE", 'finsecure',
                              [[6098, 6099], [6119, 6120], [6161, 6162], [6182, 6183], [6203, 6204]]))

# taste for active travel & adventure
qlist.append(OpinionQuestion("I ENJOY TAKING RISKS", 'enjoyrisk',
                              [[4607, 4608], [4684, 4685], [4838, 4839], [4915, 4916], [4992, 4993]]))
qlist.append(OpinionQuestion("I DO SOME SPORT/EXERCISE ONCE A WEEK", 'sportsweek',
                              [[4623, 4624], [4700, 4701], [4854, 4855], [4931, 4932], [5008, 5009]]))
qlist.append(OpinionQuestion("VEHICLE HANDLE VERY ROUGH TERRAIN IMPNT", 'alltervehicle',
                              [[3623, 3624], [3659, 3660], [3731, 3732], [3767, 3768], [3803, 3804]]))
qlist.append(OpinionQuestion("I ENJOY EATING FOREIGN FOODS", 'frgnfood',
                              [[4284,4285], [4331, 4332], [4425, 4426], [4472, 4473], [4519, 4520]]))
qlist.append(OpinionQuestion("I AM INTERESTED IN OTHER CULTURES", 'frgnculture',
                              [[4644, 4645], [4721, 4722], [4875, 4876], [4952, 4953], [5029, 5030]]))

survey = Survey('FA15_Data.txt', qlist, 'utf8', verbose=True)
survey.evaluate_questions()
scaled_df = survey.get_final_values()
#scaled_df.dropna(inplace=True)
#scaled_df = scaled_df[~(scaled_df == 0).any(axis=1)] // optionally remove any rows with zero values
scaled_df.info()


# exploratory factor analysis
fe = FactorExplore(df=scaled_df, rotation='varimax', method='principal', n_factors=2, impute='drop', verbose=True)
fe.get_barlett_sphericity()
fe.get_kmo()
ev = fe.get_eigenvalues()
fe.scree_plot(ev)
fe.get_factor_loadings()
fe.get_communalities()
factor_df = fe.get_transformed_data(scaled_df)

# define cluster drivers
cluster_list = list()
cluster_list.append(CategoryQuestion("CRUISE SHP VCATION-NUMBER TAKN LST 3 YRS", 'numcruise3yr',
                              [[24222, 24223], [24223, 24224], [24224, 24225]], [3, 2, 1]))
cluster_list.append(CategoryQuestion("FOREIGN TRAV-TOTAL #ROUND TRIPS LST 3 YR", 'frgntrav3yr',
                              [[24548, 24549], [24549, 24550], [24550, 24551], [24551, 24552]], [4, 3, 2, 1]))
cluster_list.append(OpinionQuestion("WORTH PAYING EXTRA FOR QUALITY GOODS", 'qualgoods',
                             [[4640, 4641], [4717, 4718], [4871, 4872], [4948, 4949], [5025, 5026]]))
cluster_list.append(OpinionQuestion("LIKE TO PURSUE CHALLENGE,NOVELTY,CHANGE", 'pursuechng',
                              [[4673, 4674], [4750, 4751], [4904, 4905], [4981, 4982], [5058, 5059]]))

cluster_survey = Survey('FA15_Data.txt', cluster_list, 'utf8', verbose=True)
cluster_survey.evaluate_questions()
cluster_df = cluster_survey.get_final_values()
cluster_df = pd.concat([factor_df, cluster_df], axis=1)

# combining and outputtng factors and cluster drivers
#cluster_df_scale = pd.DataFrame(preprocessing.scale(cluster_df))
cluster_df_scale = pd.DataFrame(cluster_df)
cluster_df_scale.to_csv(r'cluster_drivers.csv')

# cluster analysis
ce = ClusterExplore()
ce.plot_elbow(cluster_df_scale)
#ce.get_silhouette(cluster_df_scale)
clusters = ce.get_cluster_assignments(cluster_df, n_clusters= 3)

a = 1


# qlist.append(OpinionQuestion("I'M 1ST OF FRNDS HAVE NEW ELCTRNC EQUIP", 'ftech',
#                               [[6944, 6945], [6961, 6962], [6995, 6996], [7012, 7013], [7029, 7030]]))
# qlist.append(OpinionQuestion("PAY ANYTHING FOR ELCTRNC PROD I WANT", 'anyprice',
#                               [[6945, 6946], [6962, 6963], [6996, 6997], [7013, 7014], [7030, 7031]]))
# qlist.append(OpinionQuestion("I TRY KEEP UP/DEVELOPMENTS IN TECHNOLOGY", 'keepup',
#                               [[6952, 6953], [6969, 6970], [7003, 7004], [7020, 7021], [7037, 7038]]))
# qlist.append(OpinionQuestion("LOVE TO BUY NEW GADGETS AND APPLIANCES", 'lovenew',
#                               [[6953, 6954], [6970, 6971], [7004, 7005], [7021, 7022], [7038, 7039]]))
#
# qlist.append(OpinionQuestion("FRIENDSHIPS WOULDN'T BE CLOSE W/O CELL", 'cellfriend',
#                               [[3851, 3852], [3875, 3876], [3923, 3924], [3947, 3948], [3971, 3972]]))
# qlist.append(OpinionQuestion("MY CELL PHONE CONNECTS TO SOCIAL WORLD", 'cellsocial',
#                               [[3856, 3859], [3880, 3881], [3928, 3929], [3952, 3953], [3976, 3977]]))
# qlist.append(OpinionQuestion("CELL PHONE IS AN EXPRESSION OF WHO I AM", 'cellexpress',
#                               [[3859, 3860], [3883, 3884], [3931, 3932], [3955, 3956], [3979, 3980]]))
# qlist.append(OpinionQuestion("I LIKE TO BE CONNECTED TO FRIENDS/FAMILY", 'connectfriends',
#                               [[3866, 3867], [3890, 3891], [3938, 3939], [3962, 3963], [3986, 3987]]))



