function initalize()
{
    return Promise.all([
        require("./ScheduleModels")
            .schedule.initalize(),
        require("./UserModel").initalize(),
        require("./Vocabulary").initalize()
    ])
}

module.exports = initalize;