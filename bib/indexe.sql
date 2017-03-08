CREATE INDEX idx_d_id ON DETECTIONS(ID);
CREATE INDEX idx_fc_camid ON FRAME_CONTAINER(CAM_ID);
CREATE INDEX idx_f_ts ON FRAME(TIMESTAMP);

CREATE INDEX idx_f_fid ON FRAME(FRAME_ID);
CREATE INDEX idx_fc_fcid ON FRAME_CONTAINER(FC_ID);

CREATE INDEX idx_d_fid ON DETECTIONS(FRAME_ID);
CREATE INDEX idx_f_fcid ON FRAME(FC_ID);