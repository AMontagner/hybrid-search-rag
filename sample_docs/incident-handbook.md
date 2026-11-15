# Production incident response

For a production outage, the on-call engineer declares an incident in the incident channel and assigns an incident commander. The commander coordinates mitigation and sends a status update every 30 minutes. Roll back the latest deployment if it is the suspected cause.

After resolution, write a blameless postmortem within two business days. Include a timeline, customer impact, root cause, and action items with owners.

# Access policy

Production access requires multi-factor authentication and manager approval. Access is reviewed quarterly. Never paste credentials into tickets or chat.
