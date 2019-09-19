import pandas as pd
import sklearn
from factor_analyzer import FactorAnalyzer, Rotator
from sklearn.decomposition import FactorAnalysis
import matplotlib.pyplot as plt


class Question:
    """ support for 5-answer survey questions from fixed-width file """
    def __init__(self, question_txt, question_prefix, agree_lot_rng,
                 agree_little_rng, neither_rng, dis_little_rng, dis_lot_rng):
        """Create a Question object by passing in the location of the columns and other metadata."""
        self.question_txt = question_txt
        self.question_prefix = question_prefix
        self.agree_lot_rng = agree_lot_rng
        self.agree_little_rng = agree_little_rng
        self.neither_rng = neither_rng
        self.dis_little_rng = dis_little_rng
        self.dis_lot_rng = dis_lot_rng

        self.df_question = pd.DataFrame

    def get_column_names(self):
        """Helper function to return column names for 5-answer survey."""
        names = []
        names.append(self.question_prefix + "_agree_lot")
        names.append(self.question_prefix + "_agree_little")
        names.append(self.question_prefix + "_neither")
        names.append(self.question_prefix + "_dis_little")
        names.append(self.question_prefix + "_dis_lot")
        return names

    def load_question(self, master_df, verbose=True):
        """Loads a df of columns containing question answers. Requires the master_df to contain the same column names"""
        if verbose:
            print(f'\nLoading values for question {self.question_txt}')
        self.df_question = master_df[self.get_column_names()]

    def get_question_fequency(self):
        """Print the number of times each answer appears for a question (for validation purposes)."""
        df_temp = self.df_question.copy()
        df_temp['total'] = df_temp.sum(axis=1)  # total column
        print(f'\nFrequency of {self.question_prefix} columns:\n {df_temp[df_temp == 1].count()}')

    def count_answers(self):
        """Count how many times each answer was selected (for validation purposes)."""
        df_temp = self.df_question.copy()
        df_temp['total'] = df_temp.sum(axis=1)  # total column
        print(f'\nNumber of check boxes marked for {self.question_prefix} columns:\n {df_temp["total"].value_counts()}')

    def create_scaled_value(self):
        """Convert five column answers into a single 1-5 value."""
        df_temp = self.df_question.copy()
        df_temp.iloc[:, 0].replace(1, 5, inplace=True)
        df_temp.iloc[:, 1].replace(1, 4, inplace=True)
        df_temp.iloc[:, 2].replace(1, 3, inplace=True)
        df_temp.iloc[:, 3].replace(1, 2, inplace=True)
        df_temp[self.question_prefix] = df_temp.iloc[:, 0:5].apply(max, axis=1)
        print(f'\nFrequency of {self.question_prefix} after 1-5 scale:\n {df_temp[self.question_prefix].value_counts()}')
        return df_temp[self.question_prefix]

    def evaluate_question(self):
        """Perform validation and summarization of the question."""
        self.get_question_fequency()
        self.count_answers()
        #self.create_scaled_value()


class Survey:
    """Reads in 5-answer servey questions from a fixed-width file and evaluates/processes"""
    def __init__(self, file, question_list, encoding, read_file=True, load_questions=True, verbose=True):
        """File, questions, and encoding for the Survey"""
        self.file = file
        self.question_list = question_list,
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
        for question in question_list:
            colspecs.append(question.agree_lot_rng)
            colspecs.append(question.agree_little_rng)
            colspecs.append(question.neither_rng)
            colspecs.append(question.dis_little_rng)
            colspecs.append(question.dis_lot_rng)
            names.extend(question.get_column_names())

        self.data = pd.read_fwf(self.file, colspecs=colspecs, encoding=self.encoding, names=names, header=None)
        self.data.fillna(0, inplace=True)
        self.data = self.data.astype(int)
        return self.data

    def load_questions(self, verbose=True):
        """Loads any questions of interest in the survey."""
        for question in question_list:
            question.load_question(self.data, verbose)

    def evaluate_questions(self):
        """Evaluates questions in the survey for frequency, selected answers."""
        for question in question_list:
            question.evaluate_question()

    def get_scaled_values(self):
        """Convert questions into a single 1-5 value."""
        temp_scaled_val = []
        for question in question_list:
            temp_scaled_val.append(question.create_scaled_value())
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
        from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
        chi_square_value, p_value = calculate_bartlett_sphericity(scaled_df)
        if self.verbose:
            print(f'Bartlett Sphericity chi square value: {chi_square_value}\n')
            print(f'Baretlett Sphericity p-value: {p_value}')
        return chi_square_value, p_value

    def get_kmo(self):
        """ Check KMO """
        from factor_analyzer.factor_analyzer import calculate_kmo
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
        plt.scatter(range(1,9), ev)
        plt.plot(range(1,9), ev)
        plt.title("Scree Plot")
        plt.xlabel("Factors")
        plt.ylabel("Eigenvalue")
        plt.grid()
        plt.show()

    def get_transformed_data(self, df):
        temp_df = pd.DataFrame(self.fa.transform(df))
        return temp_df

question_list = list()
question_list.append(Question("I'M 1ST OF FRNDS HAVE NEW ELCTRNC EQUIP", 'ftech',
                              [6945, 6946], [6962, 6963], [6996, 6997], [7013, 7014], [7030, 7031]))
question_list.append(Question("PAY ANYTHING FOR ELCTRNC PROD I WANT", 'anyprice',
                              [6946, 6947], [6963, 6964], [6997, 6998], [7014, 7015], [7031, 7032]))
question_list.append(Question("I TRY KEEP UP/DEVELOPMENTS IN TECHNOLOGY", 'keepup',
                              [6953, 6954], [6970, 6971], [7004, 7005], [7021, 7022], [7038, 7039]))
question_list.append(Question("LOVE TO BUY NEW GADGETS AND APPLIANCES", 'lovenew',
                              [6954, 6955], [6971, 6972], [7005, 7006], [7022, 7023], [7039, 7040]))

question_list.append(Question("FRIENDSHIPS WOULDN'T BE CLOSE W/O CELL", 'cellfriend',
                              [3852, 3853], [3876, 3877], [3924, 3925], [3948, 3949], [3972, 3973]))
question_list.append(Question("MY CELL PHONE CONNECTS TO SOCIAL WORLD", 'cellsocial',
                              [3857, 3858], [3881, 3882], [3929, 3930], [3953, 3954], [3977, 3978]))
question_list.append(Question("CELL PHONE IS AN EXPRESSION OF WHO I AM", 'cellexpress',
                              [3860, 3861], [3884, 3885], [3932, 3933], [3956, 3957], [3980, 3981]))
question_list.append(Question("I LIKE TO BE CONNECTED TO FRIENDS/FAMILY", 'connectfriends',
                              [3867, 3868], [3891, 3892], [3939, 3940], [3963, 3964], [3987, 3988]))


survey = Survey('FA15_Data.txt', question_list, 'utf8', verbose=True)
survey.evaluate_questions()

# get scaled values and drop missing value rows
scaled_df = survey.get_scaled_values()
#scaled_df.dropna(inplace=True)
scaled_df.info()

fe = FactorExplore(df=scaled_df, rotation='varimax', method='principal', n_factors=2, impute='drop', verbose=True)
fe.get_barlett_sphericity()
fe.get_kmo()
ev = fe.get_eigenvalues()
fe.scree_plot(ev)
fe.get_factor_loadings()
fe.get_communalities()
trans_df = fe.get_transformed_data(scaled_df)



b = 1

# factor_loadings = pd.DataFrame(fa.loadings_).set_index(scaled_df.columns)
# print(f'Factor Loadings\n {factor_loadings}')

# print(f'Communalities\n {pd.DataFrame(fa.get_communalities())}\n')
# ev, v = fa.get_eigenvalues()
# print(f'Eigenvalues {ev}\n')

# plt.scatter(range(1,9), ev)
# plt.plot(range(1,9), ev)
# plt.title("Scree Plot")
# plt.xlabel("Factors")
# plt.ylabel("Eigenvalue")
# plt.grid()
# plt.show()

# factor = FactorAnalysis().fit(scaled_df)
# print(factor.compnonents_)

# fa1 = FactorAnalyzer(rotation=None)
# fa1.fit(scaled_df)
# rotator = Rotator()
# rotator.fit_transform(fa1.loadings_)