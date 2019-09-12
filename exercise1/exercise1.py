import pandas as pd

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

    def load_question(self, master_df):
        """Loads a df of columns containing question answers. Requires the master_df to contain the same column names"""
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
        df_temp['scale_value'] = df_temp.iloc[:, 0:5].apply(max, axis=1)
        print(f'\nFrequency of {self.question_prefix} after 1-5 scale:\n {df_temp["scale_value"].value_counts()}')
        return df_temp

    def evaluate_question(self):
        """Perform validation and summarization of the question."""
        self.get_question_fequency()
        self.count_answers()
        self.create_scaled_value()


class Survey:
    """Reads in 5-answer servey questions from a fixed-width file and evaluates/processes them."""
    def __init__(self, file, question_list, encoding):
        """File, questions, and encoding for the Survey"""
        self.file = file
        self.question_list = question_list,
        self.encoding = encoding

        self.data = pd.DataFrame

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

    def process_questions(self):
        """loads and evaluates any questions of interest in the survey."""
        for question in question_list:
            question.load_question(self.data)
            question.evaluate_question()


question_list = list()
question_list.append(Question("I'M 1ST OF FRNDS HAVE NEW ELCTRNC EQUIP", 'ftech',
                              [6945, 6946], [6962, 6963], [6996, 6997], [7013, 7014], [7030, 7031]))
question_list.append(Question('CMPTRS CONFUSE ME,NEVER GET USED TO THEM', 'compconf',
                              [6950, 6951], [6967, 6968], [7001, 7002], [7018, 7019], [7035, 7036]))

exercise1 = Survey('FA15_Data.txt', question_list, 'utf8')
exercise1.read_file()
exercise1.process_questions()
