import pickle
import re
import string

import contractions
import pandas as pd
import unidecode
from giskard import Dataset, Model, scan, testing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

train = pd.read_csv("./data/raw/train.csv")
test = pd.read_csv("./data/raw/test.csv")

# Drop the duplicates from the dataframe
train = train.drop_duplicates(subset=["text", "target"]).reset_index(drop=True)

# After manually going through the tweets with different target values
# Assign the target values to the tweets to find the duplicates
non_disaster = [4253, 4182, 3212, 4249, 6535, 1190, 4239, 3936, 1214, 6018]
disaster = [4193, 2803, 4554, 4250, 1207, 4317, 620, 5573]
train.loc[non_disaster, "target"] = 0
train.loc[disaster, "target"] = 1

# Again drop the duplicates from the dataframe
train = train.drop_duplicates(subset=["text", "target"]).reset_index(drop=True)

train_df = train[["text", "target"]]

train_df = train_df.iloc[:5]

wrapped_dataset = Dataset(df=train_df, target="target")  # Ground truth variable

EMOTICONS = {
    ":‑\)": "Happy face or smiley",
    ":\)": "Happy face or smiley",
    ":-\]": "Happy face or smiley",
    ":\]": "Happy face or smiley",
    ":-3": "Happy face smiley",
    ":3": "Happy face smiley",
    ":->": "Happy face smiley",
    ":>": "Happy face smiley",
    "8-\)": "Happy face smiley",
    ":o\)": "Happy face smiley",
    ":-\}": "Happy face smiley",
    ":\}": "Happy face smiley",
    ":-\)": "Happy face smiley",
    ":c\)": "Happy face smiley",
    ":\^\)": "Happy face smiley",
    "=\]": "Happy face smiley",
    "=\)": "Happy face smiley",
    ":‑D": "Laughing, big grin or laugh with glasses",
    ":D": "Laughing, big grin or laugh with glasses",
    "8‑D": "Laughing, big grin or laugh with glasses",
    "8D": "Laughing, big grin or laugh with glasses",
    "X‑D": "Laughing, big grin or laugh with glasses",
    "XD": "Laughing, big grin or laugh with glasses",
    "=D": "Laughing, big grin or laugh with glasses",
    "=3": "Laughing, big grin or laugh with glasses",
    "B\^D": "Laughing, big grin or laugh with glasses",
    ":-\)\)": "Very happy",
    ":‑\(": "Frown, sad, andry or pouting",
    ":-\(": "Frown, sad, andry or pouting",
    ":\(": "Frown, sad, andry or pouting",
    ":‑c": "Frown, sad, andry or pouting",
    ":c": "Frown, sad, andry or pouting",
    ":‑<": "Frown, sad, andry or pouting",
    ":<": "Frown, sad, andry or pouting",
    ":‑\[": "Frown, sad, andry or pouting",
    ":\[": "Frown, sad, andry or pouting",
    ":-\|\|": "Frown, sad, andry or pouting",
    ">:\[": "Frown, sad, andry or pouting",
    ":\{": "Frown, sad, andry or pouting",
    ":@": "Frown, sad, andry or pouting",
    ">:\(": "Frown, sad, andry or pouting",
    ":'‑\(": "Crying",
    ":'\(": "Crying",
    ":'‑\)": "Tears of happiness",
    ":'\)": "Tears of happiness",
    "D‑':": "Horror",
    "D:<": "Disgust",
    "D:": "Sadness",
    "D8": "Great dismay",
    "D;": "Great dismay",
    "D=": "Great dismay",
    "DX": "Great dismay",
    ":‑O": "Surprise",
    ":O": "Surprise",
    ":‑o": "Surprise",
    ":o": "Surprise",
    ":-0": "Shock",
    "8‑0": "Yawn",
    ">:O": "Yawn",
    ":-\*": "Kiss",
    ":\*": "Kiss",
    ":X": "Kiss",
    ";‑\)": "Wink or smirk",
    ";\)": "Wink or smirk",
    "\*-\)": "Wink or smirk",
    "\*\)": "Wink or smirk",
    ";‑\]": "Wink or smirk",
    ";\]": "Wink or smirk",
    ";\^\)": "Wink or smirk",
    ":‑,": "Wink or smirk",
    ";D": "Wink or smirk",
    ":‑P": "Tongue sticking out, cheeky, playful or blowing a raspberry",
    ":P": "Tongue sticking out, cheeky, playful or blowing a raspberry",
    "X‑P": "Tongue sticking out, cheeky, playful or blowing a raspberry",
    "XP": "Tongue sticking out, cheeky, playful or blowing a raspberry",
    ":‑Þ": "Tongue sticking out, cheeky, playful or blowing a raspberry",
    ":Þ": "Tongue sticking out, cheeky, playful or blowing a raspberry",
    ":b": "Tongue sticking out, cheeky, playful or blowing a raspberry",
    "d:": "Tongue sticking out, cheeky, playful or blowing a raspberry",
    "=p": "Tongue sticking out, cheeky, playful or blowing a raspberry",
    ">:P": "Tongue sticking out, cheeky, playful or blowing a raspberry",
    ":‑/": "Skeptical, annoyed, undecided, uneasy or hesitant",
    ":/": "Skeptical, annoyed, undecided, uneasy or hesitant",
    ":-[.]": "Skeptical, annoyed, undecided, uneasy or hesitant",
    ">:[(\\\)]": "Skeptical, annoyed, undecided, uneasy or hesitant",
    ">:/": "Skeptical, annoyed, undecided, uneasy or hesitant",
    ":[(\\\)]": "Skeptical, annoyed, undecided, uneasy or hesitant",
    "=/": "Skeptical, annoyed, undecided, uneasy or hesitant",
    "=[(\\\)]": "Skeptical, annoyed, undecided, uneasy or hesitant",
    ":L": "Skeptical, annoyed, undecided, uneasy or hesitant",
    "=L": "Skeptical, annoyed, undecided, uneasy or hesitant",
    ":S": "Skeptical, annoyed, undecided, uneasy or hesitant",
    ":‑\|": "Straight face",
    ":\|": "Straight face",
    ":$": "Embarrassed or blushing",
    ":‑x": "Sealed lips or wearing braces or tongue-tied",
    ":x": "Sealed lips or wearing braces or tongue-tied",
    ":‑#": "Sealed lips or wearing braces or tongue-tied",
    ":#": "Sealed lips or wearing braces or tongue-tied",
    ":‑&": "Sealed lips or wearing braces or tongue-tied",
    ":&": "Sealed lips or wearing braces or tongue-tied",
    "O:‑\)": "Angel, saint or innocent",
    "O:\)": "Angel, saint or innocent",
    "0:‑3": "Angel, saint or innocent",
    "0:3": "Angel, saint or innocent",
    "0:‑\)": "Angel, saint or innocent",
    "0:\)": "Angel, saint or innocent",
    ":‑b": "Tongue sticking out, cheeky, playful or blowing a raspberry",
    "0;\^\)": "Angel, saint or innocent",
    ">:‑\)": "Evil or devilish",
    ">:\)": "Evil or devilish",
    "\}:‑\)": "Evil or devilish",
    "\}:\)": "Evil or devilish",
    "3:‑\)": "Evil or devilish",
    "3:\)": "Evil or devilish",
    ">;\)": "Evil or devilish",
    "\|;‑\)": "Cool",
    "\|‑O": "Bored",
    ":‑J": "Tongue-in-cheek",
    "#‑\)": "Party all night",
    "%‑\)": "Drunk or confused",
    "%\)": "Drunk or confused",
    ":-###..": "Being sick",
    ":###..": "Being sick",
    "<:‑\|": "Dump",
    "\(>_<\)": "Troubled",
    "\(>_<\)>": "Troubled",
    "\(';'\)": "Baby",
    "\(\^\^>``": "Nervous or Embarrassed or Troubled or Shy or Sweat drop",
    "\(\^_\^;\)": "Nervous or Embarrassed or Troubled or Shy or Sweat drop",
    "\(-_-;\)": "Nervous or Embarrassed or Troubled or Shy or Sweat drop",
    "\(~_~;\) \(・\.・;\)": "Nervous or Embarrassed or Troubled or Shy or Sweat drop",
    "\(-_-\)zzz": "Sleeping",
    "\(\^_-\)": "Wink",
    "\(\(\+_\+\)\)": "Confused",
    "\(\+o\+\)": "Confused",
    "\(o\|o\)": "Ultraman",
    "\^_\^": "Joyful",
    "\(\^_\^\)/": "Joyful",
    "\(\^O\^\)／": "Joyful",
    "\(\^o\^\)／": "Joyful",
    "\(__\)": "Kowtow as a sign of respect, or dogeza for apology",
    "_\(\._\.\)_": "Kowtow as a sign of respect, or dogeza for apology",
    "<\(_ _\)>": "Kowtow as a sign of respect, or dogeza for apology",
    "<m\(__\)m>": "Kowtow as a sign of respect, or dogeza for apology",
    "m\(__\)m": "Kowtow as a sign of respect, or dogeza for apology",
    "m\(_ _\)m": "Kowtow as a sign of respect, or dogeza for apology",
    "\('_'\)": "Sad or Crying",
    "\(/_;\)": "Sad or Crying",
    "\(T_T\) \(;_;\)": "Sad or Crying",
    "\(;_;": "Sad of Crying",
    "\(;_:\)": "Sad or Crying",
    "\(;O;\)": "Sad or Crying",
    "\(:_;\)": "Sad or Crying",
    "\(ToT\)": "Sad or Crying",
    ";_;": "Sad or Crying",
    ";-;": "Sad or Crying",
    ";n;": "Sad or Crying",
    ";;": "Sad or Crying",
    "Q\.Q": "Sad or Crying",
    "T\.T": "Sad or Crying",
    "QQ": "Sad or Crying",
    "Q_Q": "Sad or Crying",
    "\(-\.-\)": "Shame",
    "\(-_-\)": "Shame",
    "\(一一\)": "Shame",
    "\(；一_一\)": "Shame",
    "\(=_=\)": "Tired",
    "\(=\^\·\^=\)": "cat",
    "\(=\^\·\·\^=\)": "cat",
    "=_\^= ": "cat",
    "\(\.\.\)": "Looking down",
    "\(\._\.\)": "Looking down",
    "\^m\^": "Giggling with hand covering mouth",
    "\(\・\・?": "Confusion",
    "\(?_?\)": "Confusion",
    ">\^_\^<": "Normal Laugh",
    "<\^!\^>": "Normal Laugh",
    "\^/\^": "Normal Laugh",
    "\（\*\^_\^\*）": "Normal Laugh",
    "\(\^<\^\) \(\^\.\^\)": "Normal Laugh",
    "\(^\^\)": "Normal Laugh",
    "\(\^\.\^\)": "Normal Laugh",
    "\(\^_\^\.\)": "Normal Laugh",
    "\(\^_\^\)": "Normal Laugh",
    "\(\^\^\)": "Normal Laugh",
    "\(\^J\^\)": "Normal Laugh",
    "\(\*\^\.\^\*\)": "Normal Laugh",
    "\(\^—\^\）": "Normal Laugh",
    "\(#\^\.\^#\)": "Normal Laugh",
    "\（\^—\^\）": "Waving",
    "\(;_;\)/~~~": "Waving",
    "\(\^\.\^\)/~~~": "Waving",
    "\(-_-\)/~~~ \($\·\·\)/~~~": "Waving",
    "\(T_T\)/~~~": "Waving",
    "\(ToT\)/~~~": "Waving",
    "\(\*\^0\^\*\)": "Excited",
    "\(\*_\*\)": "Amazed",
    "\(\*_\*;": "Amazed",
    "\(\+_\+\) \(@_@\)": "Amazed",
    "\(\*\^\^\)v": "Laughing,Cheerful",
    "\(\^_\^\)v": "Laughing,Cheerful",
    "\(\(d[-_-]b\)\)": "Headphones,Listening to music",
    '\(-"-\)': "Worried",
    "\(ーー;\)": "Worried",
    "\(\^0_0\^\)": "Eyeglasses",
    "\(\＾ｖ\＾\)": "Happy",
    "\(\＾ｕ\＾\)": "Happy",
    "\(\^\)o\(\^\)": "Happy",
    "\(\^O\^\)": "Happy",
    "\(\^o\^\)": "Happy",
    "\)\^o\^\(": "Happy",
    ":O o_O": "Surprised",
    "o_0": "Surprised",
    "o\.O": "Surpised",
    "\(o\.o\)": "Surprised",
    "oO": "Surprised",
    "\(\*￣m￣\)": "Dissatisfied",
    "\(‘A`\)": "Snubbed or Deflated",
}


def clean_text(text):
    """
    Clean Text
    Preprocess the given text by removing noise, special characters, URLs, etc.

    Args:
        text (str): Input text to be cleaned.

    Returns:
        str: Cleaned and preprocessed text.
    """
    # Convert the text to lowercase
    text = text.lower()

    # Remove HTML entities and special characters
    text = re.sub(r"(&amp;|&lt;|&gt;|\n|\t)", " ", text)

    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)  # remove urls

    # Remove email addresses
    text = re.sub(r"\S+@\S+", " ", text)

    # Remove dates in various formats (e.g., DD-MM-YYYY, MM/DD/YY)
    text = re.sub(r"\d{1,2}(st|nd|rd|th)?[-./]\d{1,2}[-./]\d{2,4}", " ", text)

    # Remove month-day-year patterns (e.g., Jan 1st, 2022)
    pattern = re.compile(
        r"(\d{1,2})?(st|nd|rd|th)?[-./,]?\s?(of)?\s?([J|j]an(uary)?|[F|f]eb(ruary)?|[Mm]ar(ch)?|[Aa]pr(il)?|[Mm]ay|[Jj]un(e)?|[Jj]ul(y)?|[Aa]ug(ust)?|[Ss]ep(tember)?|[Oo]ct(ober)?|[Nn]ov(ember)?|[Dd]ec(ember)?)\s?(\d{1,2})?(st|nd|rd|th)?\s?[-./,]?\s?(\d{2,4})?"
    )
    text = pattern.sub(r" ", text)

    # Remove emoticons
    emoticons_pattern = re.compile("(" + "|".join(emo for emo in EMOTICONS) + ")")
    text = emoticons_pattern.sub(r" ", text)

    # Remove mentions (@) and hashtags (#)
    text = re.sub(r"(@\S+|#\S+)", " ", text)

    # Fix contractions (e.g., "I'm" becomes "I am")
    text = contractions.fix(text)

    # Remove punctuation
    PUNCTUATIONS = string.punctuation
    text = text.translate(str.maketrans("", "", PUNCTUATIONS))

    # Remove unicode
    text = unidecode.unidecode(text)

    # Replace multiple whitespaces with a single space
    text = re.sub(r"\s+", " ", text)

    return text


params = {"solver": "liblinear", "penalty": "l2", "C": 1.0}

with open("model.pkl", "rb") as f:
    pipeline = pickle.load(f)


def prediction_function(df):
    # The pre-processor can be a pipeline of one-hot encoding, imputer, scaler, etc.
    df["cleaned_text"] = df.text.apply(lambda x: clean_text(x))
    return pipeline.predict_proba(df["cleaned_text"])


wrapped_model = Model(
    model=prediction_function,
    model_type="classification",
    classification_labels=[
        0,
        1,
    ],  # Their order MUST be identical to the prediction_function's output order
    feature_names=["text"],  # Default: all columns of your dataset
)

# By following previous user guides, you will be shown how to use your own model and dataset.
# For example purposes, we will use the demo model and dataset.
# wrapped_model = Model(model=model, model_type="classification")

scan_results = scan(wrapped_model, wrapped_dataset)

result_df = scan_results.to_dataframe()
result_df.to_csv("scan_results.csv")

# test_suite = scan_results.generate_test_suite("My first test suite")

# You can run the test suite locally to verify that it reproduces the issues
# test_suite.run()

test_suite = scan_results.generate_test_suite("My first test suite")
test_result = test_suite.add_test(
    testing.test_accuracy(wrapped_model, wrapped_dataset, threshold=0.75)
).run()

if scan_results.has_issues():
    print("Your model has vulnerabilities")
    # exit(1)
else:
    print("Your model is safe")
    # exit(0)

output = dict()
for idx, test_result in enumerate(test_result.results):
    output[idx] = {
        "Test": test_result[0],
        "Status": test_result[1].passed,
        "Threshold": test_result[2]["threshold"],
        "Score": test_result[1].metric,
    }

# for test_result in test_result.results:
#     print(f"Test {test_result[0]}\nStatus: {test_result[1].passed}")
#     print("Threshold: ", test_result[2]["threshold"])
#     print("Score: ", test_result[1].metric)
#     print("------------------")

print(output)
