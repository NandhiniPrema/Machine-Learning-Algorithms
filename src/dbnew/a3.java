package dbnew;



import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.evaluation.NominalPrediction;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.functions.SMO;
import weka.classifiers.rules.DecisionTable;
import weka.classifiers.rules.PART;
import weka.classifiers.trees.DecisionStump;
import weka.classifiers.trees.J48;
import weka.core.FastVector;
import weka.core.Instances;
 
public class a3 {
	//Reading the file by using bufferedReader//
	public static BufferedReader readDataFile(String filename) {
		BufferedReader input = null;
 //use try catch statement// 
		try {
			input= new BufferedReader(new FileReader(filename));
		} catch (FileNotFoundException ex) {
			System.err.println("File not found: " + filename);
		}
 
		return input;
	}
 // Building the classifier using trainset
	public static Evaluation classifier(Classifier model,
			Instances trainSet, Instances testSet) throws Exception {
		Evaluation evaluation = new Evaluation(trainSet);
 
		model.buildClassifier(trainSet);
		evaluation.evaluateModel(model, testSet);
 
		return evaluation;
	}
 // calculating the accuracy by using nominal value of actual and predicted value//
	public static double AccuracyCalculation(FastVector noofpredictions) {
		double correctpred = 0;
 
		for (int i = 0; i < noofpredictions.size(); i++) {
			NominalPrediction nominalpred = (NominalPrediction) noofpredictions.elementAt(i);
			if (nominalpred.predicted() == nominalpred.actual()) {
				correctpred++;
			}
		}
 
		return 100 * correctpred / noofpredictions.size();
	}
 // applying cross validation method with number of folds //
	public static Instances[][] crossValidationSplit(Instances data, int noofFolds) {
		Instances[][] splitdata = new Instances[2][noofFolds];
 
		for (int i = 0; i < noofFolds; i++) {
			splitdata[0][i] = data.trainCV(noofFolds, i);
			splitdata[1][i] = data.testCV(noofFolds, i);
		}
 
		return splitdata;
	}
 //main method//
	public static void main(String[] args) throws Exception {
		BufferedReader filedata = readDataFile("Data1new.arff");
 
		Instances data = new Instances(filedata);
		//reads the data until the last attribute//
		data.setClassIndex(data.numAttributes() - 1);
 
		// cross validation split 
		Instances[][] splitCV = crossValidationSplit(data,5);
 
		// seperating those split into the train and test set
		Instances[] trainSplits = splitCV[0];
		Instances[] testSplits = splitCV[1];
 
		//set of classifiers are applied
		Classifier[] EvaluationModels = { 
				new SMO(), // Support vector machine
			//new MultilayerPerceptron(), //
			
				//new NaiveBayes(),//decision table
				
		};
 
		//this runs for each model
		for (int j = 0; j < EvaluationModels.length; j++) {
 
			// collecting all the predictions for current model 
			FastVector totalpredictions = new FastVector();
 
			//train and test the classiifer for all the train and test split data
			for (int i = 0; i < trainSplits.length; i++) {
				Evaluation Evalvalidation = classifier(EvaluationModels[j], trainSplits[i], testSplits[i]);
 
				totalpredictions.appendElements(Evalvalidation.predictions());
 
				// Summary for each model.
				System.out.println(EvaluationModels[j].toString());
				// evaluation details
				System.out.println(Evalvalidation.toSummaryString("Evaluation results:  \n", false));
//				System.out.println("Precision = "+validation.precision(1));
//				System.out.println("Recall = "+validation.recall(1));
				System.out.println("Correct % = "+Evalvalidation.pctCorrect());
				System.out.println("Incorrect % = "+Evalvalidation.pctIncorrect());
				System.out.println("");
//				• toMatrixString – outputs the confusion matrix.
                System.out.println("Confusion Matrix:"+Evalvalidation.toMatrixString());
//				• toClassDetailsString – outputs TP/FP rates, precision, recall, F-measure,
//				AUC (per class).
                System.out.println("Class Details:"+Evalvalidation.toClassDetailsString());
//				• toCumulativeMarginDistributionString – outputs the cumulative margins
//				distribution.
			// System.out.println("cumulative margins distribution:"+validation.toCumulativeMarginDistributionString());
			   	}
 
			// Cmpute the accuracy of current classifier for all split data
			double accuracyofmodel = AccuracyCalculation(totalpredictions);
 
			// Print current classifier model name and number of accuracy predicted
			
			System.out.println("Accuracy of " + EvaluationModels[j].getClass().getSimpleName() + ": "
					+ String.format("%.2f%%", accuracyofmodel)
					+ "\n---------------------------------");
			
		}
 
	}
}

