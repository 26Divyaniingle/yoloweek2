# Annotation Guidelines — Road Object Detection

1. Draw a tight bounding box around the visible extent of each object.
2. Use the exact class ID defined in the dataset configuration.
3. Annotate all clearly identifiable objects belonging to the supported classes.
4. For partially occluded objects, box the visible portion according to the chosen annotation policy and apply that policy consistently.
5. For objects cut off by image boundaries, keep coordinates within the image.
6. Do not label shadows, reflections, road markings or background structures as objects.
7. Do not guess the class of an object that is too ambiguous to identify.
8. Check crowded scenes carefully for missing objects and overlapping boxes.
9. Do not create zero-area or negative-width/height boxes.
10. Review the saved annotations visually against the original images before accepting them.

## Quality-control pass

- Review class assignments.
- Check tightness of bounding boxes.
- Look for missing or duplicated annotations.
- Verify class IDs and normalized coordinates.
- Record corrected files and the number of issues found.
