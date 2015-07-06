Repo for my Insight Data Science project to model pedestrian risk in SF
[Insight Data Science Homepage](http://www.insightdatascience.com) 

The goal of this project is to use regression to create a predictive model of risk at intersections in San Francisco.
Previous publicly available studies (e.g. [this paper](http://docs.trb.org/prp/12-4224.pdf)) use simply (# injuries)/(ped. volume) to estimate risk.
This doesn't account for the high variability across intersections, so I sought to make a more general model.

My slides for my Insight presentation can be found here:
https://www.slideshare.net/secret/jDplGCo5TR9BBC

My data sources are:
[DataSF](http://datasf.org/) - Injuries and some infrastructure
[TransBase](http://transbasesf.org/transbase/) - Ped. traffic and infrastructure
Vehicle traffic data came from contact at SFCTA