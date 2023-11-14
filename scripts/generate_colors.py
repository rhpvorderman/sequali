"""
This creates a diverging colormap for quality score values.
The following choices were made:
- The colormap should be diverging from a center since some quality scores
  are good, others are bad and some are somewhat okay.
- The "bad" color should be red, as that is convention.
- The "good" color should be blue. Green is convention for "good", but terrible
  for colorblind people if red is used for bad.
- The middling color should be white as that is conventional for a neutral
  color.
- The "middling" quality category was chosen to be 12-15. This has less than
  one in ten errors, which is quite okay, but not stellar. Categories that
  hold quality values that stoop below the one in ten boundary are considered
  "bad".
- The above was inspired by matplotlib, yet initially a homebrew colormap was
  used that followed the guides. However this created some issues with
  signalling where the worst categories were darker and thus stood out less.
  This was pointed out to me by Marcel Martin wo pointed me to
  "Choosing colormaps"
  (https://matplotlib.org/stable/users/explain/colors/colormaps.html) by
  the matplotlib authors. The  RdBu colormap seemed to fit all the above
  criteria while also being more visually clear and being recommended as a
  good option.
"""

import matplotlib

if __name__ == "__main__":
    qualities = range(0, 47, 4)
    normalizer = matplotlib.colors.TwoSlopeNorm(12, 0, 44)
    normalized_qualities = [normalizer(q) for q in qualities]
    colormap = matplotlib.colormaps.get_cmap("RdBu")
    color_tuples = [colormap(v) for v in normalized_qualities]
    color_hexes = [matplotlib.colors.to_hex(c, keep_alpha=True) for c in color_tuples]
    print(color_hexes)
