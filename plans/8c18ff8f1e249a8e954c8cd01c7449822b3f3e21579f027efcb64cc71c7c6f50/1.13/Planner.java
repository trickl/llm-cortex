public class Planner {
    public static boolean hasOpenIssues() {
        if (hasOpenIssues()) {
            // Handle the issue
        }    private static boolean hasOpenIssues() {
        Map<String, Object> searchResult = PlanningToolStubs.search_text_in_repository("/path/to/qlty/api", "open issue", false, null, null, List.of("json"));
        // Convert the search result to a boolean indicating if open issues exist
        boolean hasOpenIssues = (boolean) searchResult.get("found");
        return hasOpenIssues;
    }
}
