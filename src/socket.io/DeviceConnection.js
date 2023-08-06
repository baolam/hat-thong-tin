const { schedule, EVENT_TYPES } = require("../models/ScheduleModels");
const { info } = require("../config/logger");

class DeviceConnection {
    constructor(device)
    {
        this.onPlanRunning();
        this.onPlanDestroyed();
        device.on("connection", (_socket) => {
            
        });
    }

    onPlanRunning()
    {
        schedule.emitter.addListener(EVENT_TYPES.RUNNING, (_info) => {
           info(`Running job with name is ${_info.name}`);
        });
    }

    onPlanDestroyed()
    {
        schedule.emitter.addListener(EVENT_TYPES.DESTROYED, (_info) => {
            info(`Destroyed job with name is ${_info.name}`);
        });
    }
}

module.exports = DeviceConnection;