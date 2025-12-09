public class Planner {
import java.util.List;public class Planner {
    public static void main(String[] args) {
        if (hasOpenIssues()) {
            // Handle the issue
        } else {
            System.out.println("No open issues found.");
        }
    }
    private static boolean hasOpenIssues() {
        Map<String, Object> result = PlanningToolStubs.search_text_in_repository("/path/to/qlty/api", "open issue", false, null, null, List.of("json"));
        return (boolean) result.get("found");
    }
}
