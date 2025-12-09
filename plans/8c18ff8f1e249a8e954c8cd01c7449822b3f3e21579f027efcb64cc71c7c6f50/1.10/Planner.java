public class Planner {
import java.util.List;public class Planner {
    public static void main(String[] args) {
        if (hasOpenIssues()) {
            // Handle the issue
        }
    }

    private static boolean hasOpenIssues() {
        // Search for open issues in the Qlty platform
        Map<String, Object> searchResult = PlanningToolStubs.search_text_in_repository("/path/to/qlty/api", "open issue", false, null, null, List.of("json"));

        // Check if any results were found
        boolean hasOpenIssues = (boolean) searchResult.get("found");

        return hasOpenIssues;
    }
}
