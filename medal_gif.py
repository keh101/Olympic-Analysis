import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
import matplotlib.pyplot as plt
import cartopy.feature as cfeature
import seaborn as sns
import pandas as pd
import imageio
import os


class OlympicMedalGif:
    def __init__(self, start_year=None, end_year=None, duration=None):
        """
        Initialize some parameters.
        :param start_year(int): The starting year of a period to show the gif. Default is 1896.
        :param end_year(int): The ending year of a period to show the gif. Default is 2018.
        :param duration(float): Set this value to tune the gif frame delay time. The larger the slower. Default is 2.5.
        """
        if start_year is not None:
            assert start_year <= 2018, 'Starting year should be no larger than 2018.'
            assert isinstance(start_year, int), 'Starting year should be an int.'
            self.start_year = start_year
        else:
            self.start_year = 1896
        if end_year is not None:
            assert isinstance(end_year, int), 'Ending year should be an int'
            self.end_year = end_year
        else:
            self.end_year = 2018
        if duration is not None:
            assert duration > 0, 'Duration should be larger than 0.'
            self.duration = duration
        else:
            self.duration = 2.5
        self.df_summer = pd.read_csv('./olympic-games/summer.csv')
        self.df_winter = pd.read_csv('./olympic-games/winter.csv')
        self.df_2016_summer = pd.read_csv('./olympic-games/summer_2016.csv')
        self.df_2018_winter = pd.read_csv('./olympic-games/winter_2018.csv')
        self.years_summer = list(dict(self.df_summer.groupby(self.df_summer['Year']).size()).keys()) + [2016]
        self.years_winter = list(dict(self.df_winter.groupby(self.df_winter['Year']).size()).keys()) + [2018]
        self.conv = pd.read_csv('country_code_convert.csv')

    def summer(self):
        """
        Create a gif describing the total medal distribution of Summer Olympics around world in a period as set from init.
        The results are stored in ./medal_figures_summer in the same parent directory.
        Most resent year is 2016.
        :return: None
        """
        assert self.start_year <= 2016, 'For summer Olympics starting year cannot be larger than 2016.'
        os.mkdir('./medal_figures_summer')
        years = [i for i in self.years_summer if (i>=self.start_year) and (i<=self.end_year)]
        cmap = sns.cubehelix_palette(n_colors=6, start=1, rot=0.1, hue=2, dark=0.3, light=1, as_cmap=True)
        shapename = 'admin_0_countries'
        countries_shp = shpreader.natural_earth(resolution='110m', category='cultural', name=shapename)
        filenames = []
        for i in years:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(1, 1, 1, projection=ccrs.Mercator())
            ax.set_extent([-169.95, 169.95, -65, 80], crs=ccrs.PlateCarree())
            ax.add_feature(cfeature.BORDERS)
            ax.coastlines(resolution='110m')
            if i == 1936:
                fig.suptitle('The Nazi Olympics', y=0.9, fontsize=14, fontweight='bold')
            elif i == 1980:
                fig.suptitle('66 Countries boycotted Soviet\'s Invasion of Afghanistan', y=0.9, fontsize=14, fontweight='bold')
            elif i == 1984:
                fig.suptitle('USSR led a boycott', y=0.9, fontsize=14, fontweight='bold')
            elif i == 1992:
                fig.suptitle('USSR has collapsed. 7 of 15 former Soviet Republics competed together as Unified Team.',
                             y=0.9, fontsize=11, fontweight='bold')
            elif i == 1896:
                fig.suptitle('The First Modern Olympic Games.', y=0.9, fontsize=14, fontweight='bold')
            elif i == 1920:
                fig.suptitle('After World War I.', y=0.9, fontsize=14, fontweight='bold')
            elif i == 1948:
                fig.suptitle('After World War II.', y=0.9, fontsize=14, fontweight='bold')

            iso_lib = list(self.conv['ISO'])
            if i != 2016:
                city = self.df_summer.loc[self.df_summer['Year'] == i]['City'].iloc[0]
                ax.title.set_text('Total Number of Medals of Summer Olympics  Year: %d  City: %s' % (i, city))
                df_tmp = self.df_summer.loc[self.df_summer['Year'] == i]
                d = dict(df_tmp.groupby(df_tmp['Country']).size())
            else:
                ax.title.set_text('Total Number of Medals of Summer Olympics  Year: %d  City: %s' % (i, 'Rio de Janeiro'))
                m = []
                for j in self.df_2016_summer['NOC'].tolist():
                    n = j[j.find('(')+1:j.find(')')]
                    m.append(n)
                k = self.df_2016_summer['Total'].tolist()
                d = dict(zip(m, k))
                d.pop('86 NOCs', None)
            max_medal = float(max(d.values()))
            for country in shpreader.Reader(countries_shp).records():
                iso = country.attributes['ADM0_A3']
                medal_num = 0
                if iso in iso_lib:
                    ioc = self.conv.loc[self.conv['ISO'] == iso,'IOC'].iloc[0]
                    if not pd.isna(ioc):
                        if ioc in d.keys():
                            medal_num = d[ioc]
                if all([iso == 'RUS', i>=1952, i<=1988, i!=1984]):
                    medal_num = d['URS']
                if all([iso=='RUS', i>=1908, i<=1912]):
                    medal_num = d['RU1']
                if all([iso=='DEU', i>=1968, i<=1988, i!=1980, i!=1984]):
                    medal_num = d['FRG'] + d['GDR']
                if all([iso=='DEU', i>=1956, i<=1964]):
                    medal_num = d['EUA']
                if i==1980 and iso=='DEU':
                    medal_num = d['GDR']
                if i==1984 and iso=='DEU':
                    medal_num = d['FRG']
                if i==1992 and iso=='RUS':
                    medal_num = d['EUN']
                ax.add_geometries(country.geometry, ccrs.PlateCarree(),
                                  facecolor=cmap(medal_num/max_medal,1))
            sm = plt.cm.ScalarMappable(cmap=cmap,norm=plt.Normalize(0,max_medal))
            sm._A = []
            plt.colorbar(sm,ax=ax, orientation="horizontal",fraction=0.046, pad=0.04)
            fname = './medal_figures_summer/year_%d.png' % i
            filenames.append(fname)
            plt.savefig(fname=fname, format='png')
            plt.close(fig)

        images = []
        for filename in filenames:
            images.append(imageio.imread(filename))
        imageio.mimsave('./medal_figures_summer/movie.gif', images, duration=self.duration)
        return

    def winter(self):
        """
        Create a gif describing the total medal distribution of Winter Olympics around world in a period as set from init.
        The results are stored in ./medal_figures_winter in the same parent directory.
        Most resent year is 2018.
        :return:
        """
        os.mkdir('./medal_figures_winter')
        start = self.start_year
        end = self.end_year
        duration = self.duration
        years = [i for i in self.years_winter if (i >= start) and (i <= end)]
        cmap = sns.cubehelix_palette(n_colors=6, start=2.5, rot=0.1, hue=2, dark=0.3, light=1, as_cmap=True)
        shapename = 'admin_0_countries'
        countries_shp = shpreader.natural_earth(resolution='110m', category='cultural', name=shapename)
        filenames = []
        for i in years:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(1, 1, 1, projection=ccrs.Mercator())
            ax.set_extent([-169.95, 169.95, -65, 80], crs=ccrs.PlateCarree())
            ax.add_feature(cfeature.BORDERS)
            ax.coastlines(resolution='110m')
            if i == 1924:
                fig.suptitle('The First Winter Olympics.', y=0.9, fontsize=14, fontweight='bold')
            if i == 1994:
                fig.suptitle('The International Olympic Committee voted to separate the Summer and Winter Games.',
                             y=0.9, fontsize=12, fontweight='bold')
            if i == 2018:
                fig.suptitle('Suspension of the Russian Olympic Committee due to Olympic Doping Controversy.',
                             y=0.9, fontsize=12, fontweight='bold')
            iso_lib = list(self.conv['ISO'])
            if i != 2018:
                city = self.df_winter.loc[self.df_winter['Year'] == i]['City'].iloc[0]
                ax.title.set_text('Total Number of Medals of Winter Olympics  Year: %d  City: %s' % (i, city))
                df_tmp = self.df_winter.loc[self.df_winter['Year'] == i]
                d = dict(df_tmp.groupby(df_tmp['Country']).size())
            else:
                ax.title.set_text('Total Number of Medals of Winter Olympics  Year: %d  City: %s' % (i, 'Pyeongchang'))
                m = []
                for j in self.df_2018_winter['NOC'].tolist():
                    n = j[j.find('(')+1:j.find(')')]
                    m.append(n)
                k = self.df_2018_winter['Total'].tolist()
                d = dict(zip(m, k))
                d.pop('30 NOCs', None)
            max_medal = float(max(d.values()))
            for country in shpreader.Reader(countries_shp).records():
                iso = country.attributes['ADM0_A3']
                medal_num = 0
                if iso in iso_lib:
                    ioc = self.conv.loc[self.conv['ISO'] == iso,'IOC'].iloc[0]
                    if not pd.isna(ioc):
                        if ioc in d.keys():
                            medal_num = d[ioc]
                if all([iso == 'RUS', i>=1956, i<=1988]):
                    medal_num = d['URS']
                if all([iso=='DEU', i>=1968, i<=1988]):
                    medal_num = d['FRG'] + d['GDR']
                if all([iso=='DEU', i>=1956, i<=1964]):
                    medal_num = d['EUA']
                if i==1952 and iso=='DEU':
                    medal_num = d['FRG']
                if i==1992 and iso=='RUS':
                    medal_num = d['EUN']
                if i==2018 and iso=='RUS':
                    medal_num = d['OAR']
                ax.add_geometries(country.geometry, ccrs.PlateCarree(),
                                  facecolor=cmap(medal_num / max_medal, 1))
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, max_medal))
            sm._A = []
            plt.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.046, pad=0.04)
            fname = './medal_figures_winter/year_%d.png' % i
            filenames.append(fname)
            plt.savefig(fname=fname, format='png')
            plt.close(fig)
        images = []
        for filename in filenames:
            images.append(imageio.imread(filename))
        imageio.mimsave('./medal_figures_winter/movie.gif', images, duration=duration)
        return

