import csv
import sys

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """
    def monthNameToNum(monthName):
        months = {"Jan": 0,
                  "Feb": 1,
                  "Mar": 2,
                  "Apr": 3,
                  "May": 4,
                  "June": 5,
                  "Jul": 6,
                  "Aug": 7,
                  "Sep": 8,
                  "Oct": 9,
                  "Nov": 10,
                  "Dec": 11,
                  }
        return months[monthName]
    evidence = []
    lables = []
    with open(filename, 'r') as dataFile:
        # create a csv reader
        dataReader = csv.reader(dataFile)
        # discard the fileds name (first row in file) 
        next(dataReader)

        for row in dataReader:
            currentUserData = []
            for i, cellVal in enumerate(row):
                # 0 - Administrative, 2 - Informational,  4- ProductRelated, 11 -OperatingSystems, 12 - Browser, 13 - Region, 14 - TrafficType
                if i in (0, 2, 4, 11, 12, 13, 14):
                    currentUserData.append(int(cellVal))
                # 10 - Month
                elif i == 10:
                    currentUserData.append(monthNameToNum(cellVal))
                # 15 - VisitorType
                elif i == 15:
                    currentUserData.append(int(cellVal == "Returning_Visitor"))
                # 16 - Weekend
                elif i == 16:
                    currentUserData.append((int(cellVal == "TRUE")))
                # 17 - Revenue
                elif i == 17:
                    lables.append((int(cellVal == "TRUE")))
                # 1 - Administrative_Duration, 3 - Informational_Duration, 5 - ProductRelated_Duration, 6 - BounceRates, 7 - ExitRates, 8 - PageValues, 9 - SpecialDay
                else:
                    currentUserData.append(float(cellVal))
            # add all current user data to evidence list
            evidence.append(currentUserData)
        return (evidence, lables)


def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    modelClassifier = KNeighborsClassifier(n_neighbors=1)
    modelClassifier.fit(evidence, labels)
    return modelClassifier
    

def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificity).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """
    truePositive = float(0)
    falseNegative = float(0)
    trueNegative = float(0)
    falsePositive = float(0)

    for prediction, lable in zip(predictions, labels):
        # True
        if prediction == lable:
            if prediction == True:
                truePositive += 1
            else:
                trueNegative += 1
        # False
        else:
            if prediction == True:
                falsePositive += 1
            else:
                falseNegative += 1

    return (truePositive/(falseNegative+truePositive), trueNegative/(falsePositive+trueNegative))


if __name__ == "__main__":
    main()
