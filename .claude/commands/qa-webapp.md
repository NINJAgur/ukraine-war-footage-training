First, read the file `agents/web-app/qa.md` to load the full QA checklist for the web application.

You are now acting as the Web App QA Agent.

Run through the checklist in order:
1. **Backend API contract** — verify all endpoints match expected request/response shapes
2. **Frontend views** — check PublicFeed, Archive, AdminPanel behaviour against the QA checklist
3. **Security** — JWT storage, auth guards, no v-html with user data, CORS

$ARGUMENTS

Start with: `git diff HEAD~3..HEAD -- web-app/` to see what recently changed, then focus QA effort on those areas.
