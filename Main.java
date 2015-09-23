import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.TreeSet;
import java.util.concurrent.TimeUnit;

/*
 * Jeremiah O'Connor
 * CSE 4334-001
 * Project 3
 */

public class Main {
	
	// constants
	static final int K_VAL = 10;
	
	// local vars
	static Map<Integer, Job> jobMap;
	static StopWords sw;
	static HashSet<String> descTerms;
	static HashSet<String> reqTerms;
	
	public static void main(String[] args) throws NumberFormatException, IOException {

		if (args.length < 2) {
			System.out.println("Invalid number of arguments");
			System.exit(0);
		}
		
		// init local vars
		jobMap = new HashMap<Integer, Job>();
		sw = new StopWords();
		descTerms = new HashSet<String>();
		reqTerms = new HashSet<String>();
		BufferedReader reader = null;
		String text = null;
		int c = 0;
		
		try {
			// load jobs file and parse job objects into the job map
			reader = new BufferedReader(new FileReader(args[0] + "jobs.tsv"));
			while ((text = reader.readLine()) != null) {
				// skip the column header line of the tsv
				if (c++ == 0)
					continue;
				
				// split vals by tab and create a new job object
				String[] vals = text.split("\\t");
				int jobId = Integer.parseInt(vals[0]);
				String desc = vals[1];
				String req = (vals.length <= 2) ? "" : vals[2];
				
				// remove all HTML tags from the strings
				desc = RemoveHTML(desc);
				req = RemoveHTML(req);
				
				Job job = new Job(jobId);
				// remove all the stop words from description and requirements
				job.DescWordList = sw.RemoveAllStopWords(desc);
				job.ReqWordList = sw.RemoveAllStopWords(req);
				
				// add all terms to the word bag for description and requirements
				descTerms.addAll(job.DescWordList);
				reqTerms.addAll(job.ReqWordList);
				
				jobMap.put(jobId, job);
			}
			reader.close();
			System.out.println("Job count: " + c);
			System.out.println("Description word bag count: " + descTerms.size());
			System.out.println("Requirements word bag count: " + reqTerms.size());
			
			long startTime = System.currentTimeMillis();
			System.out.println("\nCreating the vector space (note this calculation takes some time)...");
			
			// create doc freq counts for the df_t calculation
			// this is to reduce overhead for each idf_t calculation
			Map<String, Integer> termDocCount = new HashMap<String, Integer>();
			Map<String, Integer> reqDocCount = new HashMap<String, Integer>();
			for (String term : descTerms) {
				int docCount = 0;
				for (Job job : jobMap.values()) {
					if (job.DescWordList.contains(term))
						docCount++;
				}
				termDocCount.put(term, docCount);
			}
			for (String term : reqTerms) {
				int docCount = 0;
				for (Job job : jobMap.values()) {
					if (job.ReqWordList.contains(term))
						docCount++;
				}
				reqDocCount.put(term, docCount);
			}
			
			// used to for converting jobs hashmap into array
			// this is used to ensure consistency in the job list order since hashmaps cannot guarantee order
			Job[] jobsArr = new Job[jobMap.size()];
			c = 0;
			
			// create the weight matrix columns for each job
			// calc idf_t for each job description and requirements
			for (Job job : jobMap.values()) {
				// init tf-idf weighting column for each term
				double[] w_td = new double[descTerms.size()];
				int i = 0;
				for (String term : descTerms) {
					if (!job.DescWordList.contains(term)) {
						w_td[i] = 0;
					} else {
						// tf_t,d = number of that term t appears in document d
						double tf_td = 1.0 + Math.log10((double)Collections.frequency(job.DescWordList, term));
						// calc the idf_t for each jobs description
						// df_t = number of documents that t occurs in
						double idf_t = Math.log10(jobMap.size() / termDocCount.get(term));
						// (1 + log_10 tf_t,d) * log_10 N/df_t
						w_td[i] = tf_td * idf_t;
					}
					i++;
				}
				// save matrix weight column for description
				job.DescriptionVector = w_td;
				
				i = 0;
				w_td = new double[reqTerms.size()];
				for (String s : reqTerms) {
					if (!job.ReqWordList.contains(s)){
						w_td[i] = 0;
					} else {
						// tf_t,d = number of that term t appears in document d
						double tf_td = 1.0 + Math.log10((double)Collections.frequency(job.ReqWordList, s));
						// calc the idf_t for each jobs requirements
						// df_t = number of documents that t occurs in
						double idf_t = Math.log10(jobMap.size() / reqDocCount.get(s));
						// (1 + log_10 tf_t,d) * log_10 N/df_t
						w_td[i] = tf_td * idf_t;
					}
					i++;	
				}
				// save matrix weight column for requirements
				job.RequirementsVector = w_td;
				jobsArr[c++] = job;
				jobMap.put(job.JobId, job);
			}
			
			long endTime = System.currentTimeMillis();
			long time = (endTime - startTime);
			System.out.println(String.format("Weight matrix calculation took: %d min, %d sec\n", 
				    TimeUnit.MILLISECONDS.toMinutes(time),
				    TimeUnit.MILLISECONDS.toSeconds(time) - 
				    TimeUnit.MINUTES.toSeconds(TimeUnit.MILLISECONDS.toMinutes(time))
			));
			
			System.out.println("\nStarting k-means clustering (note this determination takes some time)...");
			startTime = System.currentTimeMillis();
			
			Object[] jobKeys = jobMap.keySet().toArray();
			// create random generator
			Random rng = new Random();
			// select k points as the initial centroids
			Cluster[] clusters = new Cluster[K_VAL];
			
			// select k points at random as the initial centroids
			int ii = 0;
			List<Integer> keyList = new ArrayList<Integer>();
			System.out.println("Initial centroid selections");
			while (ii < clusters.length) {
				int key = (Integer)jobKeys[rng.nextInt(jobKeys.length)];
				// ensure all initial centroids are unique
				if (keyList.contains(key))
					continue;
				Job job = jobMap.get(key);
				System.out.println("Cluster " + (ii+1) + ": " + job.JobId);
				// generate cluster with initial centroid
				clusters[ii] = new Cluster(job.DescriptionVector, job.RequirementsVector);
				keyList.add(key);
				ii++;
			}
			
			int iterationCount = 0;
			boolean flag = true;
			
			// k-means algorithm
			// repeat
			while (flag) {
				//	form k clusters by assigning all points to the closest centroid
				for (int i=0; i < jobsArr.length; i++) {
					int clusterNumber = 0;
					double similarity = 0;
					// compare similarities between each job and centroid
					for (int j=0; j < clusters.length; j++) {
						// sum cos sim for description and requirements vector
						double cosSim = CalculateCosineSimilarity(jobsArr[i].DescriptionVector, clusters[j].descCentroid) + 
										CalculateCosineSimilarity(jobsArr[i].RequirementsVector, clusters[j].reqCentroid);
						// store the highest similarity and cluster # for job clustering
						if (similarity < cosSim) {
							clusterNumber = j;
							similarity = cosSim;
						}
					}
					// sum and square similarities for SSE calculation
					clusters[clusterNumber].SSE += similarity * similarity;
					// add the job to the cluster with the highest similarity 
					clusters[clusterNumber].jobs.add(jobsArr[i].JobId);
				}
				// recompute the centroid of each cluster
				Cluster[] interm = new Cluster[K_VAL];
				for (int i=0; i < clusters.length; i++) {
					// create new centroid vectors for the tf_idf averages
					double[] newDescCentroid = new double[clusters[i].descCentroid.length];
					double[] newReqCentroid = new double[clusters[i].reqCentroid.length];
					// for each job in the cluster average the tf-idf values for the new centroid
					for (int jobId : clusters[i].jobs) {
						Job job = jobMap.get(jobId);
						// sum tf_idf values for each descTerm
						for (int k = 0; k < newDescCentroid.length; k++) 
							newDescCentroid[k] += job.DescriptionVector[k];
						// sum tf_idf values for each reqTerm
						for (int k = 0; k < newReqCentroid.length; k++) 
							newReqCentroid[k] += job.RequirementsVector[k];
					}
					// calc the average tf_idf for each descTerm
					for (int k=0; k < newDescCentroid.length; k++) 
						newDescCentroid[k] = newDescCentroid[k] / clusters[i].descCentroid.length;
					// calc the average tf_idf for each reqTerm
					for (int k=0; k < newReqCentroid.length; k++) 
						newReqCentroid[k] = newReqCentroid[k] / clusters[i].reqCentroid.length;
					
					// set interm centroids used for base condition
					interm[i] = new Cluster(newDescCentroid, newReqCentroid);
				}
				// until the centroids don't change
				flag = false;
				// for each cluster check if there were any changes
				for (int i=0; i < clusters.length; i++) {
					for (int k=0; k < clusters[i].descCentroid.length; k++) {
						// if a change occurred in the desc centroids then set the flag to continue iteration
						if (clusters[i].descCentroid[k] != interm[i].descCentroid[k])
							flag = true;
					}
					
					for (int k=0; k < clusters[i].reqCentroid.length; k++) {
						// if a change occurred in the req centroids then set the flag to continue iteration
						if (clusters[i].reqCentroid[k] != interm[i].reqCentroid[k])
							flag = true;
					}
				}
				
				// if there were no changes in the clusters
				// then print results
				if (!flag) {
					System.out.println("\nConvergence at " + iterationCount + " iterations!");
					endTime = System.currentTimeMillis();
					time = (endTime - startTime);
					System.out.println(String.format("k-means clustering took: %d min, %d sec\n", 
						    TimeUnit.MILLISECONDS.toMinutes(time),
						    TimeUnit.MILLISECONDS.toSeconds(time) - 
						    TimeUnit.MINUTES.toSeconds(TimeUnit.MILLISECONDS.toMinutes(time))
					));
				}
				
				// set cluster centroids to new values if convergence wasnt met
				if (flag)
					clusters = interm;
				
				iterationCount++;
			}
			
			// build output for tsv file
			List<int[]> outputList = new ArrayList<int[]>();
			int k = 0;
			for (Cluster clstr : clusters) {
				System.out.print("Custer " + ++k + ":\n");
				for (int jobId : clstr.jobs) {
					int[] outputPair = new int[2];
					outputPair[0] = jobId;
					outputPair[1] = k;
					outputList.add(outputPair);
					System.out.print(jobId + ", ");
				}
				System.out.println("\n");
			}
			
			// create output.tsv
			OutputToTSV(outputList, args[1]);
			double SSE = 0;
			// calculate the SSE for the clusters
			for (Cluster clstr : clusters) {
				SSE += clstr.SSE;
			}
			System.out.println("SSE = " + SSE);
		} catch (FileNotFoundException ex) {
			System.out.println("Error! Please review the stack trace");
			ex.printStackTrace();
		}
	}

	// remove all HTML tags
	public static String RemoveHTML(String text) {
		return text.replaceAll("\\<.*?>", "").replace("\\r"," ").replaceAll("[^a-zA-Z]", " ").replaceAll("\\s+", " ").trim();
	}
	
	// calculate the cosine similarity
	public static double CalculateCosineSimilarity(double[] v1, double[] v2){
	   	double dotProd = 0;
	   	double v1Norm = 0;
	   	double v2Norm = 0;
	   	for(int i=0; i < v1.length; i++){
	   		dotProd += v1[i] * v2[i];
	   		v1Norm += Math.pow(v1[i], 2);
	   		v2Norm += Math.pow(v2[i], 2);
	   	}
	   	v1Norm = Math.sqrt(v1Norm);
	   	v2Norm = Math.sqrt(v2Norm);
	   	return dotProd / (v1Norm * v2Norm);
	}
	
	// method for outputing the top 150 to a tsv in the format (UserID, JobID)
	public static void OutputToTSV(List<int[]> lst, String outputPath) {
		try {
		    FileWriter fos = new FileWriter(outputPath);
		    PrintWriter pr = new PrintWriter(fos);
		    for (int[] pair : lst) {
			    pr.print(pair[0] + "\t");
			    pr.print(pair[1] + "\t");
			    pr.println();
		    }
		    pr.close();
		    fos.close();
	    } catch (IOException ex) {
		    System.out.println("Error creating output.tsv. Please review the stack trace.");
		    ex.printStackTrace();
	    }
	}
}
