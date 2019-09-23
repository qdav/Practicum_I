import pandas as pd
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_kmo
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

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

    def create_final_value(self):
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


    def create_final_value(self):
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

    def create_final_value(self):
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

    def create_final_value(self):
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

    def get_scaled_values(self):
        """Convert questions into a single 1-5 value."""
        temp_scaled_val = []
        for question in self.question_list:
            test = question.create_final_value()
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


# configure/load survey and questions
qlist = list()

# qlist.append(OpinionQuestion("I'M 1ST OF FRNDS HAVE NEW ELCTRNC EQUIP", 'ftech',
#                               [[6945, 6946], [6962, 6963], [6996, 6997], [7013, 7014], [7030, 7031]]))
# qlist.append(OpinionQuestion("PAY ANYTHING FOR ELCTRNC PROD I WANT", 'anyprice',
#                               [[6946, 6947], [6963, 6964], [6997, 6998], [7014, 7015], [7031, 7032]]))
# qlist.append(OpinionQuestion("I TRY KEEP UP/DEVELOPMENTS IN TECHNOLOGY", 'keepup',
#                               [[6953, 6954], [6970, 6971], [7004, 7005], [7021, 7022], [7038, 7039]]))
# qlist.append(OpinionQuestion("LOVE TO BUY NEW GADGETS AND APPLIANCES", 'lovenew',
#                               [[6954, 6955], [6971, 6972], [7005, 7006], [7022, 7023], [7039, 7040]]))
#
# qlist.append(OpinionQuestion("FRIENDSHIPS WOULDN'T BE CLOSE W/O CELL", 'cellfriend',
#                               [[3852, 3853], [3876, 3877], [3924, 3925], [3948, 3949], [3972, 3973]]))
# qlist.append(OpinionQuestion("MY CELL PHONE CONNECTS TO SOCIAL WORLD", 'cellsocial',
#                               [[3857, 3858], [3881, 3882], [3929, 3930], [3953, 3954], [3977, 3978]]))
# qlist.append(OpinionQuestion("CELL PHONE IS AN EXPRESSION OF WHO I AM", 'cellexpress',
#                               [[3860, 3861], [3884, 3885], [3932, 3933], [3956, 3957], [3980, 3981]]))
# qlist.append(OpinionQuestion("I LIKE TO BE CONNECTED TO FRIENDS/FAMILY", 'connectfriends',
#                               [[3867, 3868], [3891, 3892], [3939, 3940], [3963, 3964], [3987, 3988]]))

# willingness to spend disposable income
qlist.append(OpinionQuestion("BUDGET ALLOWS ME TO BUY DESIGNER CLOTHES", 'designclothes',
                              [[3433, 3434], [3460, 3461], [3514, 3515], [3541, 3542], [3568, 3569]]))
qlist.append(OpinionQuestion("LIKE A NEW CAR EVERY TWO OR THREE YEARS", 'freqnewcar',
                              [[3599, 3600], [3635, 3636], [3707, 3708], [3743, 3744], [3779, 3780]]))
qlist.append(OpinionQuestion("SPEND WHAT I HAVE TO, TO LOOK YOUNGER", 'spendyounger',
                              [[4019, 4020], [4030, 4031], [4076, 4077], [4095, 4096], [4114, 4115]]))
qlist.append(OpinionQuestion("I FEEL FINANCIALLY SECURE", 'finsecure',
                              [[6099, 6100], [6120, 6121], [6162, 6163], [6183, 6184], [6204, 6205]]))


# taste for travel & adventure
qlist.append(OpinionQuestion("I ENJOY TAKING RISKS", 'enjoyrisk',
                              [[4608, 4609], [4685, 4686], [4839, 4840], [4916, 4917], [4993, 4994]]))
qlist.append(OpinionQuestion("I DO SOME SPORT/EXERCISE ONCE A WEEK", 'sportsweek',
                              [[4624, 4625], [4701, 4702], [4855, 4856], [4932, 4933], [5009, 5010]]))
# qlist.append(OpinionQuestion("VEHICLE HANDLE VERY ROUGH TERRAIN IMPNT", 'alltervehicle',
#                              [[3624, 3625], [3660, 3661], [3732, 3733], [3768, 3769], [3804, 3805]]))
qlist.append(OpinionQuestion("I ENJOY EATING FOREIGN FOODS", 'frgnfood',
                              [[4285,4286], [4332, 4333], [4426, 4427], [4473, 4474], [4520, 4521]]))
qlist.append(OpinionQuestion("I AM INTERESTED IN OTHER CULTURES", 'frgnculture',
                              [[4645, 4646], [4722, 4723], [4876, 4877], [4953, 4954], [5030, 5031]]))




# additional drivers
# income
# qlist.append(SingleValueQuestion("CRUISE SHIP VACATION-TAKEN LST 3 YRS?", 'cruise3yr', [[24199, 24200]]))
# qlist.append(SingleValueQuestion("FOREIGN TRAVEL - IN LAST 3 YEARS?", 'fgntrav3yr', [[24546, 24547]]))
# qlist.append(SingleValueQuestion("HOUSEHOLD - $100,000 OR MORE", 'house100K', [[2691, 2692]]))
#
# # where to advertise
# qlist.append(SingleValueQuestion("TRAVEL CHANNEL", 'travchannel', [[9684, 9685]]))
# qlist.append(SingleValueQuestion("NATIONAL GEOGRAPHIC CHANNEL", 'natgeo', [[9656, 9657]]))
# qlist.append(SingleValueQuestion("OUTDOOR CHANNEL", 'outdoorchanel', [[9665, 9666]]))


survey = Survey('FA15_Data.txt', qlist, 'utf8', verbose=True)
survey.evaluate_questions()
scaled_df = survey.get_scaled_values()
#scaled_df.dropna(inplace=True)
scaled_df.info()

# exploratory factor analysis
fe = FactorExplore(df=scaled_df, rotation='varimax', method='principal', n_factors=3, impute='drop', verbose=True)
fe.get_barlett_sphericity()
fe.get_kmo()
ev = fe.get_eigenvalues()
fe.scree_plot(ev)
fe.get_factor_loadings()
fe.get_communalities()
factor_df = fe.get_transformed_data(scaled_df)

# get additional drivers
cluster_list = list()
cluster_list.append(CategoryQuestion("CRUISE SHP VCATION-NUMBER TAKN LST 3 YRS", 'numcruise3yr',
                              [[24223, 24224], [24224, 24225], [24225, 24226]], [3, 2, 1]))
cluster_list.append(CategoryQuestion("FOREIGN TRAV-TOTAL #ROUND TRIPS LST 3 YR", 'frgntrav3yr',
                              [[24549, 24550], [24550, 24551], [24551, 24552], [24552, 24553]], [4, 3, 2, 1]))

cluster_survey = Survey('FA15_Data.txt', cluster_list, 'utf8', verbose=True)
cluster_survey.evaluate_questions()
cluster_df = cluster_survey.get_scaled_values()

#cluster_df = pd.concat([scaled_df, cluster_df], axis=1)

# clustering
ce = ClusterExplore()
ce.plot_elbow(cluster_df)
ce.get_silhouette(cluster_df)


