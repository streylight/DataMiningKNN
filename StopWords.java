import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;

/*
 * Jeremiah O'Connor
 * CSE 4334-001
 * Project 3
 */

// stop words object class
public class StopWords {

	// properties
	public HashSet<String> StopWordList;
	
	// ctor (see document for stop word list reference)
	public StopWords() {
		this.StopWordList = new HashSet<String>();
	    AddStopWord("a");
	    AddStopWord("able");
	    AddStopWord("about");
	    AddStopWord("above");
	    AddStopWord("according");
	    AddStopWord("accordingly");
	    AddStopWord("across");
	    AddStopWord("actually");
	    AddStopWord("after");
	    AddStopWord("afterwards");
	    AddStopWord("again");
	    AddStopWord("against");
	    AddStopWord("all");
	    AddStopWord("allow");
	    AddStopWord("allows");
	    AddStopWord("almost");
	    AddStopWord("alone");
	    AddStopWord("along");
	    AddStopWord("already");
	    AddStopWord("also");
	    AddStopWord("although");
	    AddStopWord("always");
	    AddStopWord("am");
	    AddStopWord("among");
	    AddStopWord("amongst");
	    AddStopWord("an");
	    AddStopWord("and");
	    AddStopWord("another");
	    AddStopWord("any");
	    AddStopWord("anybody");
	    AddStopWord("anyhow");
	    AddStopWord("anyone");
	    AddStopWord("anything");
	    AddStopWord("anyway");
	    AddStopWord("anyways");
	    AddStopWord("anywhere");
	    AddStopWord("apart");
	    AddStopWord("appear");
	    AddStopWord("appreciate");
	    AddStopWord("appropriate");
	    AddStopWord("are");
	    AddStopWord("around");
	    AddStopWord("as");
	    AddStopWord("aside");
	    AddStopWord("ask");
	    AddStopWord("asking");
	    AddStopWord("associated");
	    AddStopWord("at");
	    AddStopWord("available");
	    AddStopWord("away");
	    AddStopWord("awfully");
	    AddStopWord("b");
	    AddStopWord("be");
	    AddStopWord("became");
	    AddStopWord("because");
	    AddStopWord("become");
	    AddStopWord("becomes");
	    AddStopWord("becoming");
	    AddStopWord("been");
	    AddStopWord("before");
	    AddStopWord("beforehand");
	    AddStopWord("behind");
	    AddStopWord("being");
	    AddStopWord("believe");
	    AddStopWord("below");
	    AddStopWord("beside");
	    AddStopWord("besides");
	    AddStopWord("best");
	    AddStopWord("better");
	    AddStopWord("between");
	    AddStopWord("beyond");
	    AddStopWord("both");
	    AddStopWord("brief");
	    AddStopWord("but");
	    AddStopWord("by");
	    AddStopWord("c");
	    AddStopWord("came");
	    AddStopWord("can");
	    AddStopWord("cannot");
	    AddStopWord("cant");
	    AddStopWord("cause");
	    AddStopWord("causes");
	    AddStopWord("certain");
	    AddStopWord("certainly");
	    AddStopWord("changes");
	    AddStopWord("clearly");
	    AddStopWord("co");
	    AddStopWord("com");
	    AddStopWord("come");
	    AddStopWord("comes");
	    AddStopWord("concerning");
	    AddStopWord("consequently");
	    AddStopWord("consider");
	    AddStopWord("considering");
	    AddStopWord("contain");
	    AddStopWord("containing");
	    AddStopWord("contains");
	    AddStopWord("corresponding");
	    AddStopWord("could");
	    AddStopWord("course");
	    AddStopWord("currently");
	    AddStopWord("d");
	    AddStopWord("definitely");
	    AddStopWord("described");
	    AddStopWord("despite");
	    AddStopWord("did");
	    AddStopWord("different");
	    AddStopWord("do");
	    AddStopWord("does");
	    AddStopWord("doing");
	    AddStopWord("done");
	    AddStopWord("down");
	    AddStopWord("downwards");
	    AddStopWord("during");
	    AddStopWord("e");
	    AddStopWord("each");
	    AddStopWord("edu");
	    AddStopWord("eg");
	    AddStopWord("eight");
	    AddStopWord("either");
	    AddStopWord("else");
	    AddStopWord("elsewhere");
	    AddStopWord("enough");
	    AddStopWord("entirely");
	    AddStopWord("especially");
	    AddStopWord("et");
	    AddStopWord("etc");
	    AddStopWord("even");
	    AddStopWord("ever");
	    AddStopWord("every");
	    AddStopWord("everybody");
	    AddStopWord("everyone");
	    AddStopWord("everything");
	    AddStopWord("everywhere");
	    AddStopWord("ex");
	    AddStopWord("exactly");
	    AddStopWord("example");
	    AddStopWord("except");
	    AddStopWord("f");
	    AddStopWord("far");
	    AddStopWord("few");
	    AddStopWord("fifth");
	    AddStopWord("first");
	    AddStopWord("five");
	    AddStopWord("followed");
	    AddStopWord("following");
	    AddStopWord("follows");
	    AddStopWord("for");
	    AddStopWord("former");
	    AddStopWord("formerly");
	    AddStopWord("forth");
	    AddStopWord("four");
	    AddStopWord("from");
	    AddStopWord("further");
	    AddStopWord("furthermore");
	    AddStopWord("g");
	    AddStopWord("get");
	    AddStopWord("gets");
	    AddStopWord("getting");
	    AddStopWord("given");
	    AddStopWord("gives");
	    AddStopWord("go");
	    AddStopWord("goes");
	    AddStopWord("going");
	    AddStopWord("gone");
	    AddStopWord("got");
	    AddStopWord("gotten");
	    AddStopWord("greetings");
	    AddStopWord("h");
	    AddStopWord("had");
	    AddStopWord("happens");
	    AddStopWord("hardly");
	    AddStopWord("has");
	    AddStopWord("have");
	    AddStopWord("having");
	    AddStopWord("he");
	    AddStopWord("hello");
	    AddStopWord("help");
	    AddStopWord("hence");
	    AddStopWord("her");
	    AddStopWord("here");
	    AddStopWord("hereafter");
	    AddStopWord("hereby");
	    AddStopWord("herein");
	    AddStopWord("hereupon");
	    AddStopWord("hers");
	    AddStopWord("herself");
	    AddStopWord("hi");
	    AddStopWord("him");
	    AddStopWord("himself");
	    AddStopWord("his");
	    AddStopWord("hither");
	    AddStopWord("hopefully");
	    AddStopWord("how");
	    AddStopWord("howbeit");
	    AddStopWord("however");
	    AddStopWord("i");
	    AddStopWord("ie");
	    AddStopWord("if");
	    AddStopWord("ignored");
	    AddStopWord("immediate");
	    AddStopWord("in");
	    AddStopWord("inasmuch");
	    AddStopWord("inc");
	    AddStopWord("indeed");
	    AddStopWord("indicate");
	    AddStopWord("indicated");
	    AddStopWord("indicates");
	    AddStopWord("inner");
	    AddStopWord("insofar");
	    AddStopWord("instead");
	    AddStopWord("into");
	    AddStopWord("inward");
	    AddStopWord("is");
	    AddStopWord("it");
	    AddStopWord("its");
	    AddStopWord("itself");
	    AddStopWord("j");
	    AddStopWord("just");
	    AddStopWord("k");
	    AddStopWord("keep");
	    AddStopWord("keeps");
	    AddStopWord("kept");
	    AddStopWord("know");
	    AddStopWord("knows");
	    AddStopWord("known");
	    AddStopWord("l");
	    AddStopWord("last");
	    AddStopWord("lately");
	    AddStopWord("later");
	    AddStopWord("latter");
	    AddStopWord("latterly");
	    AddStopWord("least");
	    AddStopWord("less");
	    AddStopWord("lest");
	    AddStopWord("let");
	    AddStopWord("like");
	    AddStopWord("liked");
	    AddStopWord("likely");
	    AddStopWord("little");
	    AddStopWord("ll"); 
	    AddStopWord("look");
	    AddStopWord("looking");
	    AddStopWord("looks");
	    AddStopWord("ltd");
	    AddStopWord("m");
	    AddStopWord("mainly");
	    AddStopWord("many");
	    AddStopWord("may");
	    AddStopWord("maybe");
	    AddStopWord("me");
	    AddStopWord("mean");
	    AddStopWord("meanwhile");
	    AddStopWord("merely");
	    AddStopWord("might");
	    AddStopWord("more");
	    AddStopWord("moreover");
	    AddStopWord("most");
	    AddStopWord("mostly");
	    AddStopWord("much");
	    AddStopWord("must");
	    AddStopWord("my");
	    AddStopWord("myself");
	    AddStopWord("n");
	    AddStopWord("name");
	    AddStopWord("namely");
	    AddStopWord("nd");
	    AddStopWord("near");
	    AddStopWord("nearly");
	    AddStopWord("necessary");
	    AddStopWord("need");
	    AddStopWord("needs");
	    AddStopWord("neither");
	    AddStopWord("never");
	    AddStopWord("nevertheless");
	    AddStopWord("new");
	    AddStopWord("next");
	    AddStopWord("nine");
	    AddStopWord("no");
	    AddStopWord("nobody");
	    AddStopWord("non");
	    AddStopWord("none");
	    AddStopWord("noone");
	    AddStopWord("nor");
	    AddStopWord("normally");
	    AddStopWord("not");
	    AddStopWord("nothing");
	    AddStopWord("novel");
	    AddStopWord("now");
	    AddStopWord("nowhere");
	    AddStopWord("o");
	    AddStopWord("obviously");
	    AddStopWord("of");
	    AddStopWord("off");
	    AddStopWord("often");
	    AddStopWord("oh");
	    AddStopWord("ok");
	    AddStopWord("okay");
	    AddStopWord("old");
	    AddStopWord("on");
	    AddStopWord("once");
	    AddStopWord("one");
	    AddStopWord("ones");
	    AddStopWord("only");
	    AddStopWord("onto");
	    AddStopWord("or");
	    AddStopWord("other");
	    AddStopWord("others");
	    AddStopWord("otherwise");
	    AddStopWord("ought");
	    AddStopWord("our");
	    AddStopWord("ours");
	    AddStopWord("ourselves");
	    AddStopWord("out");
	    AddStopWord("outside");
	    AddStopWord("over");
	    AddStopWord("overall");
	    AddStopWord("own");
	    AddStopWord("p");
	    AddStopWord("particular");
	    AddStopWord("particularly");
	    AddStopWord("per");
	    AddStopWord("perhaps");
	    AddStopWord("placed");
	    AddStopWord("please");
	    AddStopWord("plus");
	    AddStopWord("possible");
	    AddStopWord("presumably");
	    AddStopWord("probably");
	    AddStopWord("provides");
	    AddStopWord("q");
	    AddStopWord("que");
	    AddStopWord("quite");
	    AddStopWord("qv");
	    AddStopWord("r");
	    AddStopWord("rather");
	    AddStopWord("rd");
	    AddStopWord("re");
	    AddStopWord("really");
	    AddStopWord("reasonably");
	    AddStopWord("regarding");
	    AddStopWord("regardless");
	    AddStopWord("regards");
	    AddStopWord("relatively");
	    AddStopWord("respectively");
	    AddStopWord("right");
	    AddStopWord("s");
	    AddStopWord("said");
	    AddStopWord("same");
	    AddStopWord("saw");
	    AddStopWord("say");
	    AddStopWord("saying");
	    AddStopWord("says");
	    AddStopWord("second");
	    AddStopWord("secondly");
	    AddStopWord("see");
	    AddStopWord("seeing");
	    AddStopWord("seem");
	    AddStopWord("seemed");
	    AddStopWord("seeming");
	    AddStopWord("seems");
	    AddStopWord("seen");
	    AddStopWord("self");
	    AddStopWord("selves");
	    AddStopWord("sensible");
	    AddStopWord("sent");
	    AddStopWord("serious");
	    AddStopWord("seriously");
	    AddStopWord("seven");
	    AddStopWord("several");
	    AddStopWord("shall");
	    AddStopWord("she");
	    AddStopWord("should");
	    AddStopWord("since");
	    AddStopWord("six");
	    AddStopWord("so");
	    AddStopWord("some");
	    AddStopWord("somebody");
	    AddStopWord("somehow");
	    AddStopWord("someone");
	    AddStopWord("something");
	    AddStopWord("sometime");
	    AddStopWord("sometimes");
	    AddStopWord("somewhat");
	    AddStopWord("somewhere");
	    AddStopWord("soon");
	    AddStopWord("sorry");
	    AddStopWord("specified");
	    AddStopWord("specify");
	    AddStopWord("specifying");
	    AddStopWord("still");
	    AddStopWord("sub");
	    AddStopWord("such");
	    AddStopWord("sup");
	    AddStopWord("sure");
	    AddStopWord("t");
	    AddStopWord("take");
	    AddStopWord("taken");
	    AddStopWord("tell");
	    AddStopWord("tends");
	    AddStopWord("th");
	    AddStopWord("than");
	    AddStopWord("thank");
	    AddStopWord("thanks");
	    AddStopWord("thanx");
	    AddStopWord("that");
	    AddStopWord("thats");
	    AddStopWord("the");
	    AddStopWord("their");
	    AddStopWord("theirs");
	    AddStopWord("them");
	    AddStopWord("themselves");
	    AddStopWord("then");
	    AddStopWord("thence");
	    AddStopWord("there");
	    AddStopWord("thereafter");
	    AddStopWord("thereby");
	    AddStopWord("therefore");
	    AddStopWord("therein");
	    AddStopWord("theres");
	    AddStopWord("thereupon");
	    AddStopWord("these");
	    AddStopWord("they");
	    AddStopWord("think");
	    AddStopWord("third");
	    AddStopWord("this");
	    AddStopWord("thorough");
	    AddStopWord("thoroughly");
	    AddStopWord("those");
	    AddStopWord("though");
	    AddStopWord("three");
	    AddStopWord("through");
	    AddStopWord("throughout");
	    AddStopWord("thru");
	    AddStopWord("thus");
	    AddStopWord("to");
	    AddStopWord("together");
	    AddStopWord("too");
	    AddStopWord("took");
	    AddStopWord("toward");
	    AddStopWord("towards");
	    AddStopWord("tried");
	    AddStopWord("tries");
	    AddStopWord("truly");
	    AddStopWord("try");
	    AddStopWord("trying");
	    AddStopWord("twice");
	    AddStopWord("two");
	    AddStopWord("u");
	    AddStopWord("un");
	    AddStopWord("under");
	    AddStopWord("unfortunately");
	    AddStopWord("unless");
	    AddStopWord("unlikely");
	    AddStopWord("until");
	    AddStopWord("unto");
	    AddStopWord("up");
	    AddStopWord("upon");
	    AddStopWord("us");
	    AddStopWord("use");
	    AddStopWord("used");
	    AddStopWord("useful");
	    AddStopWord("uses");
	    AddStopWord("using");
	    AddStopWord("usually");
	    AddStopWord("uucp");
	    AddStopWord("v");
	    AddStopWord("value");
	    AddStopWord("various");
	    AddStopWord("ve"); 
	    AddStopWord("very");
	    AddStopWord("via");
	    AddStopWord("viz");
	    AddStopWord("vs");
	    AddStopWord("w");
	    AddStopWord("want");
	    AddStopWord("wants");
	    AddStopWord("was");
	    AddStopWord("way");
	    AddStopWord("we");
	    AddStopWord("welcome");
	    AddStopWord("well");
	    AddStopWord("went");
	    AddStopWord("were");
	    AddStopWord("what");
	    AddStopWord("whatever");
	    AddStopWord("when");
	    AddStopWord("whence");
	    AddStopWord("whenever");
	    AddStopWord("where");
	    AddStopWord("whereafter");
	    AddStopWord("whereas");
	    AddStopWord("whereby");
	    AddStopWord("wherein");
	    AddStopWord("whereupon");
	    AddStopWord("wherever");
	    AddStopWord("whether");
	    AddStopWord("which");
	    AddStopWord("while");
	    AddStopWord("whither");
	    AddStopWord("who");
	    AddStopWord("whoever");
	    AddStopWord("whole");
	    AddStopWord("whom");
	    AddStopWord("whose");
	    AddStopWord("why");
	    AddStopWord("will");
	    AddStopWord("willing");
	    AddStopWord("wish");
	    AddStopWord("with");
	    AddStopWord("within");
	    AddStopWord("without");
	    AddStopWord("wonder");
	    AddStopWord("would");
	    AddStopWord("would");
	    AddStopWord("x");
	    AddStopWord("y");
	    AddStopWord("yes");
	    AddStopWord("yet");
	    AddStopWord("you");
	    AddStopWord("your");
	    AddStopWord("yours");
	    AddStopWord("yourself");
	    AddStopWord("yourselves");
	    AddStopWord("z");
	    AddStopWord("zero");
	    AddStopWord("nbsp");
	}
	
	// adds stop word in all lower case to the stop word list
	public void AddStopWord(String stopWord) {
		this.StopWordList.add(stopWord.toLowerCase());
	}
	
	// removes all the stop words from a word list
	public List<String> RemoveAllStopWords(String text) {
		List<String> wordList = new ArrayList<String>(Arrays.asList(text.toLowerCase().split(" ")));
		wordList.removeAll(this.StopWordList);
		return wordList;
	}
}
