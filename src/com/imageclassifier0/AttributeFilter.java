/**
 * 
 */
package com.imageclassifier0;

import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

/**
 * @author Arnaud
 *
 */

public class AttributeFilter {

	public Instances filter(Instances train_dataset) throws Exception {
		
		// Use a simple filter to remove a certain attribute
		// Here, we set up options to remove the first attribute (the filename attribute)
		String[] ops = new String[] {"-R", "1"};
		
		// Create a remove object (this is actually the filter class)
		Remove remove = new Remove();
		
		// Set the filter option
		remove.setOptions(ops);
		
		// Pass the dataset to the filter
		remove.setInputFormat(train_dataset);
		
		// Apply the filter and return new dataset
		return Filter.useFilter(train_dataset, remove);
	}
}
