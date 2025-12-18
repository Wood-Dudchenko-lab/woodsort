
import pandas as pd

def get_probe_mapping(p, print=False):

    # Returns nice dataframe with probe mapping mapping for a sanity check

    probe_df = p.to_dataframe()
    probe_df = probe_df.drop(columns=['contact_shapes', 'width', 'height'])
    probe_df = probe_df.assign(**p.contact_annotations)
    probe_df = probe_df.sort_values(
        by=['shank_ids', 'y'],
        ascending=[True, False]  # or mix True / False
    )

    if print:
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', None)

        print(probe_df)

        pd.reset_option('display.max_rows')
        pd.reset_option('display.max_columns')
        pd.reset_option('display.width')
        pd.reset_option('display.max_colwidth')

    return probe_df