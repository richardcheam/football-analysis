from sklearn.cluster import KMeans

class TeamAssigner:
    def __init__(self):
        self.team_colors = {}
        self.player_team_dict = {} #player_id : {team}

    def fit_kmeans(self, img):
        # reshape image to 2d
        img_2d = img.reshape(-1, 3)
        # peform kmeans
        kmeans = KMeans(n_clusters = 2, init = "k-means++", n_init = 1)
        kmeans.fit(img_2d)
        return kmeans

    def get_player_color(self, frame, bbox):
        img = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
        top_half_img = img[0:int(img.shape[0]/2),:]

        # cluster
        kmeans = self.fit_kmeans(top_half_img)

        # get cluster labels for each pixel
        labels = kmeans.labels_

        # reshape the labels back to original image size
        clustered_img = labels.reshape(top_half_img.shape[0], top_half_img.shape[1])

        # Get only the player cluster: [bottom left, top right, top left, bottom right]
        corner_clusters = [clustered_img[0,0], clustered_img[0,-1], clustered_img[-1,0], clustered_img[-1,-1]]
        # cluster that appears most in the corner
        non_player_cluster = max(set(corner_clusters), key = corner_clusters.count)
        player_cluster = 1 - non_player_cluster
        
        # get center which is the  [R, G, B] color
        player_color = kmeans.cluster_centers_[player_cluster]

        return player_color

    def assign_team_color(self, frame, player_detections):
        player_colors = []

        # no need track_id
        for _, player_detection in player_detections.items():
            bbox = player_detection["bbox"]
            player_color = self.get_player_color(frame, bbox)
            # append each player color to all_player_colors list
            player_colors.append(player_color)

        kmeans = KMeans(n_clusters=2, init="k-means++", n_init = 1)
        kmeans.fit(player_colors)
        # save object kmeans to save
        self.kmeans = kmeans
        # color for each team (team 1 and team 2)
        self.team_colors[1] = kmeans.cluster_centers_[0] 
        self.team_colors[2] = kmeans.cluster_centers_[1]

    def get_player_team(self, frame, player_bbox, player_id):
        # if we already assigned color just return no need to run below code
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]
        
        player_color = self.get_player_color(frame, player_bbox)

        # 0 or 1
        team_id = self.kmeans.predict(player_color.reshape(1, -1))[0]
        # convert to 1 or 0 as we have team1 and team2
        team_id += 1

        self.player_team_dict[player_id] = team_id

        return team_id

