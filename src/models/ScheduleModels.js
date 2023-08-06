
const nodecron = require("node-cron");
const shortid = require("shortid");
const events = require("events");

const Database = require("../config/database");
const sequelize = require("sequelize");

const ScheduleModel = Database.define(
    process.env.DB_SCHEDULE_TABLE,
    {
        id : {
            type : sequelize.STRING,
            allowNull : false,
            comment : "Mã code của schedule",
            primaryKey : true
        },
        name : {
            type : sequelize.TEXT,
            allowNull : false
        },
        cron_string : sequelize.STRING,
        message : sequelize.TEXT
    }
);

const EVENT_TYPES = {
    RUNNING : "RUNNING_PLAN",
    DESTROYED : "DESTROYED_PLAN"
}

class Schedule {
    emitter = new events.EventEmitter();

    async initalize()
    {
        await ScheduleModel.sync();
        return new Promise(async (resolve, reject) => {
            let list = await this.onGetList();
            if (list.length === 0)
            {
                resolve("Empty plan's list");
                return;
            }
            /// Thêm job
            list.forEach(plan => {
                this.#createPlan(plan.name, plan.cron_string)
                    .catch(err => reject(err));
            });
            resolve(`Plan's length = ${list.length}`);
        })
    }

    async onGetList()
    {
        return ScheduleModel.findAll()
            .then(vls => vls.map( vl => vl.toJSON() ))
            .catch(err => err);   
    }

    async onGetPlanBasedName(name)
    {
        return await ScheduleModel.findOne({
            where : { name }
        })
            .then(vl => {
                return { 
                    ...vl.toJSON(),
                    task : this.getTasks().get(name) 
                }
            })
    }

    /**
     * 
     * @param {*} name 
     * @param {*} cron_string 
     * @param {*} message
     * @description
     * Tạo một sự kiện lắng nghe mới 
     * @returns 
     */
    async onCreate(name, cron_string, message)
    {
        let code = nodecron.validate(cron_string);
        if (code == false)
            throw new Error(`Wrong cron_string(${cron_string})`);
        if (this.getTasks().has(name))
            throw new Error(`Job ${name} existed`);
        let id = shortid.generate();
        return new Promise((resolve, reject) => {
            ScheduleModel.create({
                id, name, message, cron_string 
            })
            .then(() => {
                this.#createPlan(name, cron_string)
                    .then(() => resolve(`Creating new plan successfully!`))
                    .catch(err => reject(err));
            })
            .catch(err => resolve(err));
        });
    }

    async onUpdateBaseName(name, wanna_update = {})
    {
        let prev_plan = await this.onGetPlanBasedName(name);
        return new Promise((resolve, reject) => {
            ScheduleModel.update(wanna_update, {
                where : { name }
            })
            .then(() => {
                if (prev_plan.cron_string != wanna_update.cron_string)
                {
                    /// Tiến hành cập nhật Cron-Jobs
                    /// Bản chất là ghi đè
                    this.#createPlan(name, wanna_update.cron_string, false);
                    resolve("Job updated!");
                }
            })
            .catch(err => reject(err));
        })
    }

    async onDeleteBaseName(name)
    {
        if (this.getTasks().has(name))
        {
            this.emitter.emit(EVENT_TYPES.DESTROYED, 
                await this.onGetPlanBasedName(name));
            this.getTasks().get(name).stop();
        }
        return ScheduleModel.destroy({
            where : { name }
        });
    }

    getTasks()
    {
        return nodecron.getTasks();
    }

    /**
     * 
     * @description
     * Hàm này được gọi khi sự kiện được thực thi
     * @param {*} name
     * @param {*} now 
     */
    #onCallback(name, now)
    {
        this.emitter.emit(EVENT_TYPES.RUNNING, 
            { name, now });
    }

    /**
     * 
     * @description
     * Thêm một kế hoạch vào cron-jobs
     * @param {*} name
     * @param {*} cron_string
     *  
     */
    async #createPlan(name, cron_string, allow_error = true)
    {
        if (this.getTasks().has(name) && allow_error)
            throw new Error(`${name} đã tồn tại!`);
        nodecron.schedule(cron_string,
            now => this.#onCallback(name, now),
            { name });
    }
}

const schedule = new Schedule();
/// THỬ HIỂN THỊ
// schedule.emitter.addListener(
//     EVENT_TYPES.RUNNING, 
//     res => console.log('Tạo mới ', res)
// );
// schedule.emitter.addListener(
//     EVENT_TYPES.DESTROYED, 
//     res => console.log('Xóa ', res)
// );

module.exports = {
    schedule,
    EVENT_TYPES
};