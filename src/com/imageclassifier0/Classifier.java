/**
 * 
 */
package com.imageclassifier0;

import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.imagefilter.AbstractImageFilter;
import weka.filters.unsupervised.instance.imagefilter.AutoColorCorrelogramFilter;
import weka.filters.unsupervised.instance.imagefilter.BinaryPatternsPyramidFilter;
import weka.filters.unsupervised.instance.imagefilter.ColorLayoutFilter;
import weka.filters.unsupervised.instance.imagefilter.EdgeHistogramFilter;

/**
 * @author Arnaud
 *
 */
public class Classifier {

	/**
	 * @param args
	 * @throws Exception 
	 */
	public static void main(String[] args) throws Exception {
		// TODO Auto-generated method stub
		
		int acc = 0;

		// The filter
		ColorLayoutFilter imFilter = new ColorLayoutFilter();
		
		// The filter that removes the filename
		AttributeFilter attributeFilter = new AttributeFilter();
		
		// Load dataset
		DataSource train_source = new DataSource("D:\\Development\\ML\\ImageClassifier0\\train_butterfly_vs_owl\\train_butterfly_vs_owl.arff");
		DataSource test_source = new DataSource("D:\\Development\\ML\\ImageClassifier0\\test_butterfly_vs_owl\\test_butterfly_vs_owl.arff");
		Instances train_dataset = train_source.getDataSet();
		Instances test_dataset = test_source.getDataSet();
		
		//System.out.println(train_dataset.toString());
		//System.out.println(test_dataset.toString());
		
		// Set the dataset to the last attributes
		train_dataset.setClassIndex(train_dataset.numAttributes() - 1);
		test_dataset.setClassIndex(train_dataset.numAttributes() - 1);
		
		// Pass the datasets to the filter
		imFilter.setInputFormat(train_dataset);
		imFilter.setInputFormat(test_dataset);
		
		// Apply the filter
		train_dataset = Filter.useFilter(train_dataset, imFilter);
		test_dataset = Filter.useFilter(test_dataset, imFilter);
		
		// Apply the filter that removes the file name
		// This is to avoid the  weka.core.UnsupportedAttributeTypeException: weka.classifiers.trees.J48: Cannot handle string attributes! error
		train_dataset = attributeFilter.filter(train_dataset);
		test_dataset = attributeFilter.filter(test_dataset);
		
		// Create and build the classifier 
		// with the decision tree algorithm
		J48 decisionTree = new J48();
		decisionTree.buildClassifier(train_dataset);
		
		// Evaluating the classifier
		for(int i = 0; i < test_dataset.numInstances(); i++) {
			//System.out.print(test_dataset.instance(i)+ " ");
			double index = decisionTree.classifyInstance(test_dataset.instance(i));
			if (index == test_dataset.instance(i).classValue()) {
				acc++;
			}
			//String className = train_dataset.classAttribute().value((int)index);
			//System.out.println(className);;
		}
		
		System.out.println("Accuracy: "+ (double)acc/test_dataset.numInstances() * 100 + "%");
	}
}
