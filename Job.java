import java.util.HashMap;
import java.util.List;
import java.util.Map;

/*
 * Jeremiah O'Connor
 * CSE 4334-001
 * Project 3
 */

// job object class
public class Job {
	
	// properties
	public int JobId;
	public List<String> DescWordList;
	public List<String> ReqWordList;
	public double[] DescriptionVector;
	public double[] RequirementsVector;
	
	// ctor
	public Job(int jobId) {
		this.JobId = jobId;
	}
}
