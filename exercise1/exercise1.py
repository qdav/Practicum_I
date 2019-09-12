import pandas as pd

class Question:
    """ support for 5-answer survey questions from fixed-width file """
    def __init__(self, question_txt, question_prefix, agree_lot_rng,
                 agree_little_rng, neither_rng, dis_little_rng, dis_lot_rng):
        """creating a Question object requires passing in the location of the column and other metadata"""
        self.question_txt = question_txt
        self.question_prefix = question_prefix
        self.agree_lot_rng = agree_lot_rng
        self.agree_little_rng = agree_little_rng
        self.neither_rng = neither_rng
        self.dis_little_rng = dis_little_rng
        self.dis_lot_rng = dis_lot_rng

        self.df_question = pd.DataFrame

    def getColumnNames(self):
        """helper function to return column names for 5-answer survey"""
        names = []
        names.append(self.question_prefix + "_agree_lot")
        names.append(self.question_prefix + "_agree_little")
        names.append(self.question_prefix + "_neither")
        names.append(self.question_prefix + "_dis_little")
        names.append(self.question_prefix + "_dis_lot")
        return names

    def loadQuestion(self, master_df):
        """loads a df of columns containg question answers
            requires the master_df to contain the same column names"""
        print(f'\nLoading values for question {self.question_txt}')
        self.df_question = master_df[self.getColumnNames()]

    def getQuestionFrequency(self):
        """print the number of times each answer appears for a question"""
        df_temp = self.df_question.copy()
        df_temp['total'] = df_temp.sum(axis=1)  # total column
        print(f'\nFrequency of {self.question_prefix} columns:\n {df_temp[df_temp == 1].count()}')

    def countAnswers(self):
        """count how many times each answer was selected (for validation purposes"""
        df_temp = self.df_question.copy()
        df_temp['total'] = df_temp.sum(axis=1)  # total column
        print(f'\nNumber of check boxes marked for {self.question_prefix} columns:\n {df_temp["total"].value_counts()}')

    def createScaledValue(self):
        """convert five column answers into a single 1-5 value"""
        df_temp = self.df_question.copy()
        df_temp.iloc[:, 0].replace(1, 5, inplace=True)
        df_temp.iloc[:, 1].replace(1, 4, inplace=True)
        df_temp.iloc[:, 2].replace(1, 3, inplace=True)
        df_temp.iloc[:, 3].replace(1, 2, inplace=True)
        df_temp['scale_value'] = df_temp.iloc[:, 0:5].apply(max, axis=1)
        print(f'\nFrequency of {self.question_prefix} after 1-5 scale:\n {df_temp["scale_value"].value_counts()}')
        return df_temp

    def evaluateQuestion(self):
        """perform validation and summarization of question"""
        self.getQuestionFrequency()
        self.countAnswers()
        self.createScaledValue()


class Survey:
    """reads in 5-answer servey questions from a fixed-width file and evaluates/processes them """
    def __init__(self, file, question_list, encoding):
        """contextual information for creating survey"""
        self.file = file
        self.question_list = question_list,
        self.encoding = encoding

        self.data = pd.DataFrame

    def readFile(self):
        """reads in survey file based on any specified question objects"""
        colspecs = [[0, 7]]  # for the id
        names = ['id']
        for question in question_list:
            colspecs.append(question.agree_lot_rng)
            colspecs.append(question.agree_little_rng)
            colspecs.append(question.neither_rng)
            colspecs.append(question.dis_little_rng)
            colspecs.append(question.dis_lot_rng)
            names.extend(question.getColumnNames())

        self.data = pd.read_fwf(self.file, colspecs=colspecs, encoding=self.encoding, names=names, header=None)
        self.data.fillna(0, inplace=True)
        self.data = self.data.astype(int)
        return self.data

    def processQuestions(self):
        """loads and evaluates any questions of interest in the survey"""
        for question in question_list:
            question.loadQuestion(self.data)
            question.evaluateQuestion()


question_list = list()
question_list.append(Question("I'M 1ST OF FRNDS HAVE NEW ELCTRNC EQUIP", 'ftech',
                              [6945, 6946], [6962, 6963], [6996, 6997], [7013, 7014], [7030, 7031]))
question_list.append(Question('CMPTRS CONFUSE ME,NEVER GET USED TO THEM', 'compconf',
                              [6950, 6951], [6967, 6968], [7001, 7002], [7018, 7019], [7035, 7036]))

exercise1 = Survey('FA15_Data.txt', question_list, 'utf8')
exercise1.readFile()
exercise1.processQuestions()
