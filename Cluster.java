import java.util.ArrayList;
import java.util.List;

/*
 * Jeremiah O'Connor
 * CSE 4334-001
 * Project 3
 */

// cluster object class
public class Cluster {

	// properties
	public double[] descCentroid;
	public double[] reqCentroid;
	public List<Integer> jobs;
	public double SSE;
	
	// ctor
	public Cluster(double[] dc, double[]rc) {
		this.descCentroid = dc;
		this.reqCentroid = rc;
		this.jobs = new ArrayList<Integer>();
	}
}
