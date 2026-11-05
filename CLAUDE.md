# Neighborhood similarity search

## Goal
Given a place I know (e.g. Rice Village, Montrose, Memorial Park in Houston), return the most similar areas in another city, restricted to a search boundary (a city or bounding box). I am deciding where to move, so output is a shortlist of areas to visit, not a verdict.

## Constraints
- Do not run heavy compute locally. Training and embedding run on hosted GPUs (Kaggle or Colab free tier; Modal or a rented GPU if quota runs out). Write code as notebooks or scripts meant for those environments.
- Download big datasets directly on the remote machine (wget), not through my laptop.
- Cache every expensive step to disk in chunks and save outputs (embeddings, checkpoints, FAISS index) to a Kaggle dataset, Hugging Face Hub, or Drive.
- Start with Houston plus one target city. Prove the pipeline end to end before scaling to more metros.

## Design decisions made so far
- Unit of analysis: ~800m walkshed cells on a uniform grid (~400m spacing), not named neighborhoods. Named neighborhoods are only a labeling layer.
- Approach: self-supervised pretraining of an area encoder (POI set + street stats, optionally review-text vectors), then nearest-neighbor retrieval with FAISS. Few-shot personalization only: with a handful of labeled places, labels tune ~10-20 feature weights and serve as an eval set. Do not train a large projection on them.
- Do not average anchors into one profile. Keep one query vector per anchor.
- Wikipedia neighborhood text does not match my own sense of these places, so it is not a primary text signal.
- Text signals to try: aggregated review text per cell (Google Local Reviews, Yelp Open Dataset) plus Reddit snippets linked to neighborhoods via a gazetteer. Test on the three anchors before wiring into training.
- Fix city leakage: center text/feature vectors per city, scrub city, street and neighborhood names from text, and run a probe that tries to predict the city from the embedding.

## Data sources
- Google Local Reviews (UCSD McAuley lab): per-state metadata and review files. Field names are from memory (gmap_id, latitude, longitude, text, num_of_reviews); print one raw row first.
- Yelp Open Dataset: one tar; Houston may not be covered (check business.json). The city field is a raw string, so suburbs split metros.
- OpenStreetMap: POIs, street graph, neighborhood `place` nodes (via osmnx; Geofabrik extracts for bulk).
- Neighborhood names/points: SimpleMaps free tier covers the 25 largest US cities (Houston yes, Austin no), points only, CC BY 4.0 with attribution. For other cities use OSM `place` nodes or Wikidata. Calibrate OSM against SimpleMaps in Houston before trusting it elsewhere.
- Geoapify cities file (OSM-derived, settlement level only) is useful for choosing cities and bounding boxes, not for neighborhoods.
- Overture Maps is a possible POI supplement.

## Evaluation
- Leave-one-out and held-out-city splits where possible.
- City-leakage probe on embeddings.
- Within-Houston sanity: an anchor's nearest Houston cells should look like the same kind of place.
- Gut-check queries in cities I know, written down before looking at results.
- Final test: visit top-ranked areas in a candidate city.
- Compare against the non-learned baseline (standardized feature distance).

## Build order
1. `01_data_coverage_check.ipynb` (exists): per-anchor and per-city grid coverage in Google Local, Yelp, OSM. Not yet run against real data.
2. Data prep: cell grid, POI sets per cell, review embeddings per cell, neighborhood labels per cell.
3. Pretraining notebook for the area encoder.
4. Query step: FAISS index per city, non-max suppression of overlapping cells, reverse-geocode to names.
5. Optional: personal tuning, then a small web app.

## Open items
- Target cities not decided. Austin is a candidate. Add them to `CITIES` (bounding boxes) in the notebook.
- Anchor coordinates in the notebook are approximate; verify on a map.
- Confirm Houston and anchor names appear in the SimpleMaps CSV (Houston confirmed present; check anchors).
- Verify Google Local download links and field names, and that the target states are covered.
