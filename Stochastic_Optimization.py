import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import os, copy, csv
from inspect import currentframe
import matplotlib.pyplot as plt

def calculate_risk(decision, riskData, weights):
    final_risk = 0
    accounted_columns = set()  # Track columns that have been accounted for
    for row, selected in enumerate(decision):
        if selected:
            for col in range(len(riskData[row])):
                if col not in accounted_columns:  # Check if column has been accounted for
                    final_risk += riskData[row][col] * weights[col]
                    accounted_columns.add(col)  # Mark column as accounted for
    return final_risk

class Base(object):
    def __init__(self,
                 target_percentage: float = 1,
                 searchPath: str = r"C:\\Users\\" + str(os.getlogin()) + "\\Desktop\\multiObjective4",
                 modelName: str = "NFMOLE",
                 minimumRows: int = 1,
                 riskIdxs: list = [],
                 realRisk: int = 0,
                 objectiveWeight: tuple = (1.0, 1.0),
                 objectiveIdx: tuple = (0, 1),
                 sensitivityThreshold: float = .2,
                 applySensitivity: bool = False,
                 keepRowIdxs: list = []) -> None:

        self.target_percentage, self.searchPath, self.modelName, \
        self.minimumRows, self.riskIdxs, self.realRisk, self.objectiveWeight, \
        self.objectiveIdx, self.sensitivityThreshold, self.applySensitivity, \
        self.keepRowIdxs = target_percentage, searchPath, modelName, minimumRows, riskIdxs, \
                           realRisk, objectiveWeight, objectiveIdx, sensitivityThreshold, applySensitivity, keepRowIdxs

    def queryFile(self,
                  fileName: str = "NFMOLE.csv") -> str:
        for root, _, files in os.walk(self.searchPath):
            return os.path.join(root, fileName) \
                if fileName in files else None

    @staticmethod
    def getLineNum() -> int:
        return currentframe().f_back.f_lineno


class ImportData(Base):

    def __init__(self,
                 *args,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def preprocess_data_and_weights(self, riskdata, weights, scenario_values):
        # Identify columns with sums >= threshold (replace 150 with your desired threshold)
        valid_columns = [i for i, value in enumerate(scenario_values) if value >= 150]

        # Filter out columns with total sum < threshold from the data
        filtered_data = [[row[i] for i in valid_columns] for row in riskdata]

        # Filter the weights using the same valid columns indices
        filtered_weights = [weights[i] for i in valid_columns]
        return filtered_data, filtered_weights

    def importRiskData(self, data_path="C:\\Users\\" + str(os.getlogin()) + "\\Desktop\\multiObjective4\\NFMOLE.csv") -> tuple[list, int]:
        """
        Import the risk data as a 2d array (list)
        [1] -> Convert the risk data to binary values based on the formula:
               value >= .05 ==> 1
               value <  .05 ==> 0
        """

        with open(data_path, 'r') as csvfile:
            reader = csv.reader(csvfile)
            data = list(reader)

        # BINARY REPRESENTATION OF RISK DATA
        riskD = [
            [1 if float(val) >= .05 else 0 for val in row] \
            for row in data
        ]

        # Weights
        weights_path = "C:\\Users\\" + str(os.getlogin()) + "\\Desktop\\multiObjective4\\weights.csv"
        with open(weights_path, 'r') as csvfile:
            reader = csv.reader(csvfile)
            weights_data = [row[0] for row in reader if row]

        weights = [float(w) for w in weights_data[:len(data[0])]]  # Convert strings to floats instead of integers
        applyWeight = [self.sensitivityThreshold for _ in weights] if self.applySensitivity else weights
        if len(data[0]) != len(weights):
            raise Exception("Risk data length does not match Weights length" + str(self.getLineNum()))

        ## VALIDATE ALL ELEMENTS IN 2D ARRAY ARE BINARY
        assert len([elem for r in range(len(riskD)) for elem in riskD[r] if elem in (1, 0)]) == len([it for item in riskD for it in item]), \
            "Risk Data is not binary!\nLine --> " + str(self.getLineNum())

        return riskD, applyWeight

    def importScenarios(self, scenario_path="C:\\Users\\" + str(os.getlogin()) + "\\Desktop\\multiObjective4\\scenarios.csv"):
        """
        Import the scenario data from a CSV file and filter scenarios based on the threshold.
        """
        with open(scenario_path, 'r') as csvfile:
            reader = csv.reader(csvfile)
            scenarios = list(reader)
        
        scenario_values = [float(row[0]) for row in scenarios]
        valid_scenarios = [i for i, value in enumerate(scenario_values) if value >= 150]

        return scenario_values, valid_scenarios

class ApplyOpt(ImportData):

    def __init__(self,
                 *args,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def calculate_target_risk(self, i, j, riskData, selected_rows):
        target_risk = 0
        while i < len(riskData):
            if not any(riskData[l][j] == 1 for l in range(i)):
                target_risk += riskData[i][j] * selected_rows[i]
            i += 1
        return target_risk

    def adjust_selected_rows_and_calculate_final_risk(self, riskData):
        adjusted_riskData = copy.deepcopy(riskData)
        num_rows = len(riskData)
        num_cols = len(riskData[0]) if riskData else 0  # Check if riskData is not empty

        for col_index in range(num_cols):
            seen_numbers = set()  # To track numbers already seen in the current column
            for row_index in range(num_rows):
                current_number = adjusted_riskData[row_index][col_index]
                if current_number in seen_numbers:
                    adjusted_riskData[row_index][col_index] = 0.0
                else:
                    seen_numbers.add(current_number)

        return adjusted_riskData

    def applySolution(self) -> None:
        results = []

        for tp in [i * 0.1 for i in range(1, 11)]:
            self.target_percentage = tp
            riskData, weights = self.importRiskData()
            scenario_values, valid_scenarios = self.importScenarios()
            riskData, weights = self.preprocess_data_and_weights(riskData, weights, scenario_values)

            scenarios = [valid_scenarios]
            expected_value = 0
            for scenario_data in scenarios:
                numRows: int = len(riskData)

                # CREATE THE MODEL
                model = gp.Model(self.modelName)

                # ADD VARIABLES
                selected_rows = model.addVars(numRows,
                                              name="selected_rows",
                                              vtype=GRB.BINARY)

                ## SET MODEL SENSE --> BY DEFAULT MINIMIZE
                targetRisk = sum(
                    self.calculate_target_risk(0, j, riskData, selected_rows)
                    for j in range(len(riskData[0]))
                )

                # SET OBJECTIVE FUNCTIONS
                model.setObjectiveN(targetRisk,
                                    index=self.objectiveIdx[0],
                                    weight=self.objectiveWeight[0],
                                    priority=1,
                                    name="maxRisk")
                model.setObjectiveN(sum(selected_rows),
                                    index=self.objectiveIdx[1],
                                    weight=-1.0 * self.objectiveWeight[1],
                                    priority=0,  # ALWAYS PRIORITIZE THE NUMBER OF ROWS SELECTED
                                    name="minRows")

                # ADD CONSTRAINTS TO THE MODEL
                for row in range(1, numRows):
                    model.addConstr(selected_rows[row] <= selected_rows[row - 1],
                                    name="incrementRows")

                model.addConstr(sum(selected_rows[row] \
                                    for row in range(numRows)) >= self.minimumRows,
                                name="min_selected_rows")

                model.addConstr(
                    targetRisk >= self.target_percentage * sum(weights),
                    name="target_percentage"
                )

                # OPTIMIZE THE MODEL
                model.optimize()

                if model.status == GRB.OPTIMAL:
                    decision = model.getAttr('x')

                    newRiskData = copy.deepcopy(riskData)
                    original_indices = list(range(len(riskData)))

                    for r in range(len(newRiskData) - 1):
                        for j in range(len(newRiskData) - r - 1):
                            if sum(newRiskData[j]) <= sum(newRiskData[j + 1]):
                                newRiskData[j], newRiskData[j + 1] = newRiskData[j + 1], newRiskData[j]
                                original_indices[j], original_indices[j + 1] = original_indices[j + 1], original_indices[j]

                    # CALCULATE FINAL RISK AFTER APPLYING THE SOLUTION
                    final_risk = calculate_risk(decision, riskData, weights)  # Use the calculate_risk function

                    final_selected_rows = [original_indices[row] for row in range(len(decision)) if decision[row]]

                    # STORE RESULTS
                    results.append({
                        'target_percentage': tp,
                        'final_risk': final_risk,
                        'selected_rows': final_selected_rows
                    })

                    print("Model is optimal.")
                    print("Target Risk:", targetRisk.getValue())
                    print("Final Selected Rows:", final_selected_rows)
                    print("Final Risk:", final_risk)

        # PLOT RESULTS
        plt.figure(figsize=(10, 6))
        target_percentages = [result['target_percentage'] for result in results]
        final_risks = [result['final_risk'] for result in results]
        plt.plot(target_percentages, final_risks, marker='o')
        plt.xlabel('Target Percentage')
        plt.ylabel('Final Risk')
        plt.title('Final Risk vs Target Percentage')
        plt.grid(True)
        plt.show()

         # Calculate the maximum risk achieved
        max_risk_achieved = max(result[2] for result in results if result[2] is not None)

        # Add the differences column to results
        results_with_diff = [(tp, num_selected, risk_achieved, max_risk_achieved - risk_achieved if risk_achieved is not None else None)
                             for (tp, num_selected, risk_achieved) in results]

        # Normalize the values
        df = pd.DataFrame(results_with_diff, columns=['target_percentage', 'num_selected_rows', 'risk_achieved', 'difference'])
        df['num_selected_rows_normalized'] = (df['num_selected_rows'] - df['num_selected_rows'].min()) / (df['num_selected_rows'].max() - df['num_selected_rows'].min())
        df['difference_normalized'] = (df['difference'] - df['difference'].min()) / (df['difference'].max() - df['difference'].min())

        # Export results to a CSV file
        df.to_csv('results.csv', index=False)

        # Plot the scatter plot
        fig, ax = plt.subplots()
        scatter = ax.scatter(df['difference_normalized'], df['num_selected_rows_normalized'])
        ax.set_xlabel('-Risk')
        ax.set_ylabel('Number of Detectors')
        plt.show()


    def getSolution(self) -> None:
        riskData, _ = self.importRiskData()
        numRows: int = len(riskData)
        for row in range(numRows):
            print(riskData[row])


if __name__ == "__main__":
    ApplyOpt().applySolution()
