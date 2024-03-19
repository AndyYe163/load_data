def load_data():
    # Competency: reading files in Pandas, df manipulation, regex
    
    # The three variables are initialized to None. You will fill them with the correct values. 
    energy, gdp, scim_en = [None] * 3
    
    # YOUR CODE HERE
    energy = pd.read_excel('assets/Energy Indicators.xls',
                           names=[
                               'Country', 'Energy Supply',
                               'Energy Supply per Capita', '% Renewable'
                           ],
                           usecols=[2, 3, 4, 5],
                           skiprows=17,
                           skipfooter=38)
    energy.replace('...', np.NAN, inplace=True)
    energy['Energy Supply'] *= 1e6
    energy['Country'] = energy.Country.apply(lambda x: re.sub('[0-9]', '', x))
    energy.replace({
        "Republic of Korea": "South Korea",
        "United States of America": "United States",
        "United Kingdom of Great Britain and Northern Ireland": "United Kingdom",
        "China, Hong Kong Special Administrative Region": "Hong Kong",  
        
    },inplace = True)
    energy['Country'] = energy['Country'].apply(lambda x: x.split(' (')[0] if "(" in x else x)
    energy.set_index('Country', inplace=True)
    
    gdp = pd.read_csv('assets/world_bank.csv', skiprows=4)
    gdp.replace({"Korea, Rep.": "South Korea", "Iran, Islamic Rep.": "Iran", "Hong Kong SAR, China": "Hong Kong"}, inplace=True)
    gdp.set_index('Country Name', inplace=True)
    gdp.rename_axis('Country', inplace=True) 
    
    scim_en = pd.read_excel('assets/scimagojr-3.xlsx')
    scim_en.set_index('Country', inplace=True)

    #raise NotImplementedError()
    
    return energy, gdp, scim_en
