function initalize()
{
    return Promise.all([
        require("./ScheduleModels")
            .schedule.initalize(),
        require("./UserModel").initalize()
    ])
}

module.exports = initalize;