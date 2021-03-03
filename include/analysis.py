import math
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from likertScalePlot import likert_scale
from textCleaning import plot_wordcloud as _plot_wordcloud
from textCleaning import wrap_clean_text


def get_sampled_df(df, columns):
    """
    Return a subset of the general df by selecting the columns and adding the
    columns of year and Country
    """
    if isinstance(columns, str):
        columns = [columns]
    initial_col = ["Country", "Year"]
    initial_col.extend(columns)
    return df[initial_col]


def count_y_n(df, category, normalize=False, dropna=True):
    """
    """
    df_sub = df.iloc[:, df.columns != "Country"]
    df_sub = df_sub.iloc[:, df_sub.columns != "Year"]
    results = df_sub.apply(
        pd.Series.value_counts, dropna=dropna, normalize=normalize
    ).reset_index()
    # results = results.transpose()
    results.columns = [category, "Count"]

    # results = results.reset_index()
    results.set_index(category, inplace=True)
    results.columns = ["Count"]
    # print(df_sub.loc['Yes'])
    results["Percentage"] = (results["Count"] / results["Count"].sum()) * 100

    results = results.reindex(["Yes", "No"])
    return results


def count_multi_column(
    df, country, order_question=False, dropna=False, normalize=False
):
    raise NotImplementedError


def count_one_choice(df, country, order_question=False, dropna=False, normalize=False):
    """
    Count the values of different columns and transpose the count
    :params:
        :df pd.df(): dataframe containing the data
    :return:
        :result_df pd.df(): dataframe with the count of each answer for each columns
    """
    df_one_count = df.iloc[:, -1].value_counts().to_frame()
    df_one_count.columns = ["Count"]

    df_one_count["Percentage"] = (
        df_one_count["Count"] / df_one_count["Count"].sum() * 100
    )

    if order_question:
        df_one_count.sort_index(inplace=True)

    return df_one_count


def count_multi_choice(df, category, dropna=True):
    """
    Count the values of different columns and transpose the count. It expect multi-choice type of answers
    :params:
        :df pd.df(): dataframe containing the data
    :return:
        :result_df pd.df(): dataframe with the count of each answer for each columns
    """
    # Subset the columns by removing the country columns
    df_sub = df.iloc[:, df.columns != "Country"]

    # Subset only the columns that have values for the country. It checks if the unique list is more than just nan.
    col_to_keep = list()
    for col in df_sub:
        if len(df_sub[col].unique()) > 1:
            col_to_keep.append(col)
    df_final = df_sub[col_to_keep]

    # As the No can be considered as absence of Yes, fill the value 'No' with na to keep Yes only
    df_final = df_final.fillna(value="No")

    # Create the total of participants that have answered 'Yes' at at least one field
    total_answered = len(df_final.notnull().any(axis=1))

    # Calculate the count for the column
    df_final = df_final.apply(pd.Series.value_counts, dropna=dropna)

    # Replace all the 0 with NA
    df_final.fillna(value=0, inplace=True)

    df_final = df_final.loc["Yes"]

    df_final = df_final.to_frame().reset_index()
    df_final.columns = [category, "Count"]

    # Calculate the proportion of each category by participants
    # that answered at least one category (rather than by total answer for all categories)
    df_final["Percentage"] = (df_final["Count"] / total_answered) * 100

    # Get the category information between Bracket
    df_final[category] = df_final[category].str.replace("]", "", regex=False).str.split("[").str[1]

    # Reorder the df
    df_final = df_final.sort_values("Percentage", ascending=False)

    # Reset the index on category
    df_final = df_final.set_index(category)

    # Return the results
    return df_final


def count_likert(df, likert_answer=False, dropna=False, normalize=False, reindex=False):
    """
    Count the values of different columns and transpose the count
    :params:
        :df pd.df(): dataframe containing the data
    :return:
        :result_df pd.df(): dataframe with the count of each answer for each columns
    """
    # df_to_use = get_sampled_df(df, columns)

    # Subset the columns by removing the country columns
    df_sub = df.loc[:, df.columns != "Country"]

    def convert_to_int(x):
        try:
            return int(x)
        except ValueError:
            return x

    # first convert the np.nan into a value that is different
    df_sub = df_sub.fillna("to_remove")

    # # then transform into a string
    df_sub = df_sub.applymap(str)

    # # then replace the -1 into np.nan
    df_sub = df_sub.replace({"to_remove": np.nan})

    # Calculate the counts for them
    df_count = df_sub.apply(pd.Series.value_counts, dropna=dropna, normalize=normalize)

    # Reorder according to the answers order found in the folder if argument is passed
    if likert_answer:
        for (
            i
        ) in (
            likert_answer
        ):  # Add the missing likert because they have nan value and are not in the dataset
            i = str(i)
            if i not in df_count.index:
                df_count.loc[i] = np.nan
        df_count = df_count.reindex(index=likert_answer)

    return df_count


def count_diff(
    df,
    columns,
    country,
    category,
    survey_year,
    multi_choice=False,
    multi_column=False,
    y_n=False,
    order_index=False,
    disable_past_year=False,
):
    """
    Check if it is possible to get the difference from previous year.
    Return a dataframe with the count of each category
    """
    # Get the count for the survey year
    survey_year_prev = survey_year - 1
    df_to_use = get_sampled_df(df, columns)
    try:
        df_to_use[columns].astype("category")
    except NotImplementedError:
        pass
    if country == "all":
        df_country_survey_year = df_to_use[(df_to_use["Year"] == survey_year)]
    else:
        df_country_survey_year = df_to_use[
            (df_to_use["Country"] == country) & (df_to_use["Year"] == survey_year)
        ]

    if multi_choice:
        count_current_field_survey_year = count_multi_choice(df_country_survey_year, category)
    elif y_n:
        count_current_field_survey_year = count_y_n(df_country_survey_year, category)
    elif multi_column:
        count_current_field_survey_year = count_multi_column(df_country_survey_year, category)
    else:
        count_current_field_survey_year = count_one_choice(df_country_survey_year, category)

    # Get the count for the previous year if it exists
    if not disable_past_year:
        if country == "all":
            df_country_survey_year_prev = df_to_use[(df_to_use["Year"] == survey_year_prev)]
        else:
            df_country_survey_year_prev = df_to_use[
                (df_to_use["Country"] == country) & (df_to_use["Year"] == survey_year_prev)
            ]

        if multi_choice is True:
            try:
                count_current_field_survey_year_prev = count_multi_choice(df_country_survey_year_prev, category)
            except KeyError:
                count_current_field_survey_year_prev = None

        elif y_n is True:
            count_current_field_survey_year_prev = count_y_n(df_country_survey_year_prev, category)

        else:
            count_current_field_survey_year_prev = count_one_choice(df_country_survey_year_prev, category)
    else:
        count_current_field_survey_year_prev = None
    # Calculate the difference
    try:
        count_current_field_survey_year[f"Percentage in {survey_year_prev}"] = count_current_field_survey_year_prev[
            "Percentage"
        ]
        count_current_field_survey_year["Difference with previous year"] = (
            count_current_field_survey_year["Percentage"]
            - count_current_field_survey_year_prev["Percentage"]
        ).to_frame()
    except TypeError:
        pass
    except Exception:
        print(count_current_field_survey_year)
        print(survey_year_prev)
        print(count_current_field_survey_year_prev)
        raise

    # Drop all columns with full na. It removes Difference if it does not exists
    count_current_field_survey_year.dropna(axis=1, how="all", inplace=True)

    # Change name of index
    col_name = "{} for {}".format(category, country)
    count_current_field_survey_year.index.name = col_name

    if order_index:
        if isinstance(order_index, list):
            count_current_field_survey_year = count_current_field_survey_year.reindex(order_index)
        else:
            count_current_field_survey_year = count_current_field_survey_year.sort_index()

    return count_current_field_survey_year


def describe_quant(df, category, remove_outliers):
    """
    """

    df = df.iloc[:, 2]
    df.dropna(inplace=True)
    if remove_outliers:
        try:
            df = df[(df < np.percentile(df, 95))]
        except IndexError:  # In case 2017 is empty
            pass

        # df.dropna(inplace=True)
    result = df.describe().to_frame()
    return result


def describe_diff(df, columns, country, category, survey_year, remove_outliers=True):
    """
    """
    df_to_use = get_sampled_df(df, columns)
    survey_year_prev = survey_year - 1

    df_country_survey_year = df_to_use[
        (df_to_use["Country"] == country) & (df_to_use["Year"] == survey_year)
    ]
    df_country_survey_year_prev = df_to_use[
        (df_to_use["Country"] == country) & (df_to_use["Year"] == survey_year_prev)
    ]
    result_survey_year = describe_quant(df_country_survey_year, category, remove_outliers)
    index_name = f"{category} for {country}" + (" (without 95 percentile)" if remove_outliers else "")
    result_survey_year.index.name = index_name
    result_survey_year.columns = [f"Results in {survey_year}"]
    result_survey_year_prev = describe_quant(df_country_survey_year_prev, category, remove_outliers)
    if result_survey_year_prev.iloc[0, 0] > 0:
        result_survey_year[f"Results in {survey_year_prev}"] = result_survey_year_prev.iloc[:, 0]

    result_survey_year.dropna(axis=1, how="all", inplace=True)

    return result_survey_year


def count_ranking(df, columns, country, category, survey_year):
    """
    Count the number of time a value appears in one columns and do that for all the columns
    provided. Assuming the format of the column name is "$code. $Question text [rank$numb]",
    it extracts the ranking number and use it as new header. It return the dataframe of the
    counted values
    :params:
        :df pd.df(): dataframe containing the data
        :country str(): which country has to be sampled
        :category str(): rename the dataframe with that category
    :return:
        :result_df pd.df(): dataframe with the count of each answer for each columns
    """
    df_to_use = get_sampled_df(df, columns)
    df_country_survey_year = df_to_use[
        (df_to_use["Country"] == country) & (df_to_use["Year"] == survey_year)
    ]
    df_one_count = df_country_survey_year.iloc[:, 2:].apply(pd.Series.value_counts)
    df_one_count.columns = [x.split("[")[1][:-1] for x in df_one_count.columns]
    df_one_count = df_one_count.apply(lambda x: x / x.sum() * 100)
    df_one_count.sort_values("Rank 1", inplace=True, ascending=False)
    df_one_count.index.name = category
    return df_one_count


# TODO: accept several list of likert
def plotting_likert(
    df, country, category, to_plots, survey_year, type_orga="horizontal", order_scale=None
):
    """"""
    # nbr_plots = len(to_plots)
    nbr_plots = 1

    if type_orga == "vertical":
        fig, axs = plt.subplots(nbr_plots, 1, sharex=True, figsize=(9, 4))
    else:
        fig, axs = plt.subplots(1, nbr_plots, figsize=(9, 4))
    list_plots = list()

    for i, one_plot in enumerate(to_plots):
        columns = to_plots
        category = category
        df_sub = get_sampled_df(df, columns=columns)

        df_country_survey_year = df_sub[
            (df_sub["Country"] == country) & (df_sub["Year"] == survey_year)
        ]
        df_country_survey_year = df_country_survey_year.drop("Year", axis="columns")
        try:
            axs[i] = likert_scale(
                count_likert(df_country_survey_year, order_scale).transpose(),
                normalise=True,
                legend=True,
                title_plot="{}: {}".format(category, country),
                ax=axs[i],
            )
            axs[i].set_title(category)
            list_plots.append(axs[i])

        except TypeError:
            axs = likert_scale(
                count_likert(df_country_survey_year, order_scale).transpose(),
                normalise=True,
                legend=True,
                title_plot="{}: {}".format(category, country),
                ax=axs,
            )
            axs.set_title(category)
            list_plots.append(axs)

    for ax in list_plots:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.invert_yaxis()


# Todo: Merge into plotting_likert()
def plotting_time_likert(
    country,
    df_time_spent=None,
    df_time_wish=None,
    df_time_diff=None,
    dict_time_diff=None,
):
    """"""
    plt.tight_layout()
    nbr_plots = len(
        [x for x in [df_time_spent, df_time_wish, df_time_diff] if x is not None]
    )
    fig, axs = plt.subplots(nbr_plots, 1, sharex=True, figsize=(10, 14))
    list_plots = list()
    legend_loc = 'lower center'
    ncol = 16

    if df_time_spent is not None:
        try:
            axs[0] = likert_scale(
                count_likert(
                    df_time_spent[df_time_spent["Country"] == country],
                    likert_answer=[str(i) for i in range(11)[1:]],
                ).transpose(),
                normalise=True,
                legend=True,
                legend_loc=legend_loc,
                legend_ncol=ncol,
                title_plot="{}: Time spent for each type of activity".format(country),
                ax=axs[0],
            )
            axs[0].set_title("Time spent")
            list_plots.append(axs[0])

        except TypeError:
            axs = likert_scale(
                count_likert(
                    df_time_spent[df_time_spent["Country"] == country],
                    likert_answer=[str(i) for i in range(11)[1:]],
                ).transpose(),
                normalise=True,
                legend=True,
                legend_loc=legend_loc,
                legend_ncol=ncol,
                title_plot="{}: Time spent for each type of activity".format(country),
                ax=axs,
            )
            axs.set_title("Time spent")
            list_plots.append(axs)

    if df_time_wish is not None:
        axs[1] = likert_scale(
            count_likert(
                df_time_wish[df_time_wish["Country"] == country],
                likert_answer=[str(i) for i in range(11)[1:]],
            ).transpose(),
            normalise=True,
            legend=True,
            legend_loc=legend_loc,
            legend_ncol=ncol,
            title_plot="{}: Time wish to spent for each type of activity".format(
                country
            ),
            ax=axs[1],
        )
        axs[1].set_title("Time wish to spent")

        list_plots.append(axs[1])

    if df_time_diff is not None:
        all_unique_diff = [
            x
            for x in np.unique(
                df_time_diff[dict_time_diff["Country"] == country]
                .loc[
                    :,
                    df_time_diff[dict_time_diff["Country"] == country].columns
                    != "Country",
                ]
                .values
            )
        ]
        negative_order = [str(x) for x in all_unique_diff if x < 0] + [
            str(x) for x in all_unique_diff if x >= 0
        ]
        axs[2] = likert_scale(
            count_likert(
                df_time_diff[dict_time_diff["Country"] == country],
                likert_answer=negative_order,
            ).transpose(),
            normalise=True,
            legend=True,
            legend_loc=legend_loc,
            legend_ncol=ncol,
            title_plot="{}: Difference between time wish to spent and actually spent for each type of activity".format(
                country
            ),
            ax=axs[2],
        )
        axs[2].set_title("Difference between time spent and wish")
        list_plots.append(axs[2])

    for ax in list_plots:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.invert_yaxis()


def plot_density_func(df, columns, category, country, survey_year, remove_outliers=True):
    survey_year_prev = survey_year - 1
    df_sampled = get_sampled_df(df, columns=columns)
    df_sampled.columns = ["Country", "Year", "Value"]
    df = df_sampled[df_sampled.Country == country]
    # Remove na
    df.dropna(inplace=True)
    # Remove the outliers
    if remove_outliers:
        df_survey_year = df[df.Year == survey_year]
        df_survey_year = df_survey_year[(df_survey_year.Value < np.percentile(df_survey_year.Value, 95))]
        df_survey_year_prev = df[df.Year == survey_year_prev]
        if len(df[df["Year"] == survey_year_prev]) > 0:

            df_survey_year_prev = df_survey_year_prev[(df_survey_year_prev.Value < np.percentile(df_survey_year_prev.Value, 95))]
            df = pd.concat([df_survey_year, df_survey_year_prev])
        else:
            df = df_survey_year
        df.dropna(inplace=True)

    fig, axarr = plt.subplots(1, 2, figsize=(8, 6))
    sns.boxplot(x="Year", y="Value", data=df, ax=axarr[0])

    sns.swarmplot(
        x="Year", y="Value", data=df, ax=axarr[0], color="grey", alpha=0.75
    )  # .set_title('{}: {}'.format(category, country))

    if len(df[df["Year"] == survey_year_prev]) > 0:
        sns.distplot(
            df[df["Year"] == survey_year_prev]["Value"].dropna(),
            bins=int(len(df[df["Year"] == survey_year_prev]) / 2),
            label=str(survey_year_prev),
            ax=axarr[1],
        )

    sns.distplot(
        df[df["Year"] == survey_year]["Value"].dropna(),
        bins=int(len(df[df["Year"] == survey_year]) / 2),
        label=str(survey_year),
        ax=axarr[1],
    )

    sns.despine(offset=10, trim=True)
    title = f"{category} for {country}" + (" (without 95 percentile)" if remove_outliers else "")
    fig.suptitle(title)
    plt.legend()


def plot_cat_comparison(df, country, category, order_index=False, width=6.4):
    # Plotting the current categories and the difference with the last year
    from matplotlib import rcParams
    rcParams.update({'figure.autolayout': True})
    plt.ioff()
    if order_index:
        if isinstance(order_index, list):
            df = df.reindex(order_index)
        else:
            df = df.sort_index()
    ind = np.arange(len(df.index))
    height = max(len(df.index) / 4, 2)

    try:
        fig, axs = plt.subplots(
            1, 2, sharey=True, figsize=(width, height), gridspec_kw={"width_ratios": [7, 1]}
        )

        plt.gcf().subplots_adjust(left=0.2)
        # current field
        axs[0].barh(ind, df["Percentage"], align="center")

        axs[0].set_title("{}, {}".format(category, country))
        rects = axs[0].patches

        # Difference from last year

        axs[1].barh(
            ind,
            df["Difference with previous year"],
            color=df["Difference with previous year"].apply(
                lambda x: "g" if x > 0 else "orange"
            ),
        )
        axs[1].set_title("Δ")

        for ax in axs:
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_visible(False)

    except KeyError:
        # Set up columns
        fig, ax = plt.subplots(figsize=(width, height))
        if order_index is False:
            df.sort_values("Percentage", ascending=False, inplace=True)

        # current field
        plt.barh(ind, df["Percentage"], align="center")
        plt.title("{}, {}".format(category, country))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)


    # For each bar: Place a label
    for rect in rects:
        # Get X and Y placement of label from rect.
        x_value = rect.get_width()
        y_value = rect.get_y() + rect.get_height() / 2

        # Number of points between bar and label
        space = 15
        # Vertical alignment for positive values
        va = "center"

        # If value of bar is negative: Place label below bar
        if x_value < 0:
            # Invert space to place label below
            space *= -1
            # Vertically align label at top

        # Use Y value as label and format number with one decimal place
        label = "{:.0f}%".format(x_value)

        # Create annotation
        try:
            axs[0].annotate(
                label,  # Use `label` as label
                (x_value, y_value),  # Place label at end of the bar
                xytext=(space, 0),  # Vertically shift label by `space`
                textcoords="offset points",  # Interpret `xytext` as offset in points
                ha="center",  # Horizontally center label
                va=va,
            )  # Vertically align label differently for
            # positive and negative values.
        except UnboundLocalError:
            ax.annotate(
                label,  # Use `label` as label
                (x_value, y_value),  # Place label at end of the bar
                xytext=(0, space),  # Vertically shift label by `space`
                textcoords="offset points",  # Interpret `xytext` as offset in points
                ha="center",  # Horizontally center label
                va=va,
            )  # Vertically align label differently for
            # positive and negative values.
    plt.yticks(ind, df.index)


def plot_ranking(df, category, country):
    fig, ax = plt.subplots(figsize=(7, 5))

    # Dropping the columns with NA (the ranks that are not present)
    df = df.dropna(axis=1, how="all")

    df.T.plot(
        kind="barh",
        title="{}: {}".format(category, country),
        grid=False,
        ax=ax,
        stacked=True,
    )

    ax.set(xlabel="Ranking of importance", ylabel="Percentage")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    # Put a legend below current axis
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=2,
    )
    for p in ax.patches:
        if int(round(p.get_width())) >= 5:
            ax.annotate(
                "{}".format(str(int(round(p.get_width())))),
                (
                    p.get_x() + p.get_width() / 2.0,
                    p.get_y() + p.get_height() / 2 - 0.1,
                ),
                ha="center",
            )


def plot_wordcloud(df, columns, country, category, survey_year):
    plt.figure()
    df_to_sample = get_sampled_df(df, columns=columns)
    df = df_to_sample[
        (df_to_sample["Country"] == country) & (df_to_sample["Year"] == survey_year)
    ]
    txt_to_plot = wrap_clean_text(df, columns)
    plot = _plot_wordcloud(txt_to_plot)
    plt.imshow(plot, cmap=plt.cm.gray, interpolation="bilinear")
    plt.axis("off")
    plt.title("{}, {}".format(category, country))


def radar_plotting(
    df, title="", subplot=False, percentage=True, fixed_y=False, color=None
):
    """
    Plotting a radar based on the df.
    The df need to have the group in the index
    and the different variable in columns.

    :params:
    --------
         :df dataframe(): data to be plotted
         :sub_plot bool(): if the different categories are plotted
         on the same plot or on different subplot. Default is false
    :return:
    --------
        plot(): of the data
    """

    def _draw_plot(df, label, fixed_y, ax=None, color=None):

        # number of variable
        categories = list(df)
        N = len(categories)
        # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
        angles = [n / float(N) * 2 * math.pi for n in range(N)]
        angles += angles[:1]
        if ax is None:
            ax = plt.subplot(111, projection="polar")

        # If you want the first axis to be on top:
        ax.set_theta_offset(math.pi / 2)
        ax.set_theta_direction(-1)

        # Draw one axe per variable + add labels labels yet
        plt.xticks(angles[:-1], categories)
        # Draw ylabels
        ax.set_rlabel_position(0)
        if fixed_y is True:
            yticks_range = list(range(10, 100, 10))
        else:
            yticks_range = list(range(int(df.min().min()), int(df.max().max()), 10))
        yticks_labels = [str(i) for i in yticks_range]

        plt.yticks(yticks_range, yticks_labels, color="grey", size=7)
        if fixed_y is True:
            plt.ylim(0, 100)
        else:
            plt.ylim(0, int(df.max().max()) + 5)
        ax.tick_params(axis="x", colors="grey")
        ax.spines["polar"].set_visible(False)
        # ax.tick_params(axis='y', colors='grey')
        # ax.yaxis.grid(False,color='grey',linestyle='-')
        # plt.ylim(0,100)
        # ------- PART 2: Add plots
        # Plot each individual = each line of the data
        for i in range(len(df)):
            values = df.iloc[i, :].values.flatten().tolist()
            values += values[:1]
            if color:
                if isinstance(color, list):
                    color_ = color[i]
                else:
                    color_ = color
                ax.plot(
                    angles,
                    values,
                    c=color_,
                    linewidth=1,
                    linestyle="solid",
                    label=df.index[i],
                )
                ax.fill(angles, values, c=color_, alpha=1 / (len(df) + 2))
            else:
                ax.plot(
                    angles, values, linewidth=1, linestyle="solid", label=df.index[i]
                )
                ax.fill(angles, values, alpha=1 / (len(df) + 2))

        # Add a title

        # Adjust position if subplot or not
        if subplot:
            position_title = 1.1
        else:
            position_title = 1.1

        if color and not isinstance(color, list):
            plt.title(label, size=14, color=color, y=position_title)
        else:
            plt.title(label, size=14, color="grey", y=position_title)
        return ax

    # Initialise the spider plot
    plt.figure()
    # Create a color palette:
    if subplot is False:
        # fig, ax = plt.subplots(1, nbr_plots, sharey=True, polar=True)
        if color:
            color_ = [matplotlib.colors.to_rgba(x, alpha=None) for x in color]
            # color_ = [matplotlib.colors.to_rgb(x) for x in color]

        _draw_plot(df=df, label=title, fixed_y=fixed_y, color=color_)

    else:
        my_palette = plt.cm.get_cmap("Set2", len(df.index))

        for i, cat in enumerate(df.index):
            ax = plt.subplot(3, 3, i + 1, projection="polar")
            df_to_plot = df.iloc[[i]]
            _draw_plot(
                df=df_to_plot, label=cat, ax=ax, color=my_palette(i), fixed_y=fixed_y
            )

    # Add legend
    if subplot is False:
        plt.legend(loc="upper right", bbox_to_anchor=(0.1, 0.1))
    plt.tight_layout()
