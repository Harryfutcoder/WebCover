# Local Deployment Configuration

Set credentials in your shell for test accounts you control. They are not saved
in the paper-entry plan. The explicit entry forwards only the documented
credential variables; it still rejects inherited experiment-setting overrides.

| Site | Environment variables |
| --- | --- |
| Nextcloud | `WEBTEST_NEXTCLOUD_USERNAME`, `WEBTEST_NEXTCLOUD_PASSWORD` |
| Realworld | `WEBTEST_REALWORLD_USERNAME`, `WEBTEST_REALWORLD_EMAIL`, `WEBTEST_REALWORLD_PASSWORD` |
| Agilefant | `WEBTEST_AGILEFANT_USERNAME`, `WEBTEST_AGILEFANT_PASSWORD` |
| 4gaBoards | `WEBTEST_4GABOARDS_USERNAME`, `WEBTEST_4GABOARDS_EMAIL`, `WEBTEST_4GABOARDS_PASSWORD` |
| TimeOff | `WEBTEST_TIMEOFF_EMAIL`, `WEBTEST_TIMEOFF_PASSWORD` |
| Gadael | `WEBTEST_GADAEL_EMAIL`, `WEBTEST_GADAEL_PASSWORD` |
| Odoo local deployment | `WEBTEST_ODOO_DB`, `WEBTEST_ODOO_LOGIN`, `WEBTEST_ODOO_PASSWORD` |
| Pagekit (WebRLED) | `WEBTEST_PAGEKIT_USERNAME`, `WEBTEST_PAGEKIT_PASSWORD` |

For QExplore, supply credentials in its local site configuration immediately
before running, and never commit those values. The tracked configuration has
empty credential fields. The runner does not automatically load `.env` files.

Configure Chrome/ChromeDriver paths in `settings.yaml` and the appropriate
external runner. Keep browser versions compatible. Set up test-site deployments,
reset procedures and the embedding/model assets separately. Source publication
does not establish that every historical launch helper is portable.

