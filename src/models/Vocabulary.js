const nodecron = require("node-cron");
const { info, logger } = require("../config/logger");

const { GlobalEvents, TYPE } = require("../config/global_event");
const Database = require("../config/database");
const sequelize = require("sequelize");

const VocabularyModel = Database.define(
    process.env.DB_WORD_TABLE,
    {
        word : {
            type : sequelize.TEXT,
            primaryKey : true,
            allowNull : false
        },
        meaning : {
            type : sequelize.TEXT,
            allowNull : false
        },
        timing : {
            type : sequelize.INTEGER,
            allowNull : false,
            comment : "Thời gian thực hiện việc ôn từ vựng [day_code]. Là mã thời gian ôn",
            unique : true
        },
        forgot_rate : {
            type : sequelize.FLOAT, // [0, 1] với giá trị càng thấp, người dùng càng ít quên
            comment : "Tỉ lệ người dùng quên đi từ"
        },
        type : {
            // Loại tử
            type : sequelize.STRING
        }
    }
);

const TYPE_WORD = {
    NOUN : "noun",
    VERB : "vern",
    UNDEF : "undefined",
    ADJ : "adj"
};

const UserModel = require("./UserModel");
class Vocabulary {
    /// Thực thi lắng nghe sự kiện qua ngày mới và cập nhật timing các từ
    constructor()
    {
        /// Danh sách các từ trong ngày hiện tại :>
        this.__progress = new Map();
        this.__user = {};
        GlobalEvents.addListener(TYPE.GET_DEFAULT_USER, () => this.#onSetUser());

        /// Cứ sau 1 ngày thì lặp lại một lần
        let date = new Date();
        let rule = `${date.getSeconds()} ${date.getMinutes()} ${date.getHours()} * * *`;
        info(
            `Building vocabulary_job running every day at ${date.getHours()}:${date.getMinutes()}:${date.getSeconds()}`
        );
        
        this.task = nodecron.schedule(rule, 
            (now) => this.#onMonitor(now), {
            name : "vocabulary_job"
        });
    }

    /**
     * 
     * @param {*} words
     * @description
     * Thủ tục này để dựng nên progress 
     * @returns 
     */
    #onBulidingProgress(words)
    {
        if (words.length === 0)
            return;
        this.__progress.clear();
        for (let i = 0; i < words.length; i++)
        {
            let { word, meaning, forgot_rate } = words[i];
            this.__progress.set(word, { meaning, forgot_rate });
        }
    }

    /**
     * 
     * @description
     * Tiến hành gán người dùng nếu chưa gán
     */
    #onSetUser()
    {
        /// Gọi thủ tục này để lấy thông tin người dùng đc cài đặt sẵn
        UserModel.onGetDefaultUser()
            .then(vl => { this.__user = vl })
            .then(() => this.#onMonitor())
            .catch((err) => logger.log("error", err));
    }

    /**
     * 
     * @description
     * Gọi thủ tục này để tiến hành xây dựng quản lí các từ cần học ở ngày hiện tại
     */
    async #onMonitor()
    {
        info(`Crawling words from ${process.env.DB_WORD_TABLE}`);
        return this.getWordsBaseDayLeft(this.__user.timing)
            .then(words => this.#onBulidingProgress(words));
    }

    /**
     * 
     * @param {*} timing
     * @description
     * Lấy danh sách từ dựa vào số ngày còn lại 
     * @returns 
     */
    async getWordsBaseDayLeft(timing)
    {
        return VocabularyModel.findAll({
            where : { timing }
        })
        .then(vls => vls.map( vl => vl.toJSON() ));
    }

    /**
     * 
     * @description
     * Kết nối đến CSDL với Table đc chỉ định
     * @returns 
     */
    async initalize()
    {
        return VocabularyModel.sync();
    }

    async onGetVocabs()
    {
        return VocabularyModel.findAll()
            .then(vls => vls.map(vl => vl.toJSON()));
    }

    /**
     * 
     * @description
     * Lấy các từ sẽ học trong ngày
     * @returns 
     */
    async onGetCurrentWords()
    {
        return new Promise((resolve, reject) => {
            if (this.__progress.size === 0)
                reject(`No word, today :>`);
            else {
                let l = [];
                this.__progress.forEach((value, key) => l.push([key, value]));
                resolve(l);
            }
        });
    }

    async getWord(word)
    {
        return VocabularyModel.findOne({
            where : { word }
        })
        .then(vl => vl.toJSON());
    }

    /**
     * 
     * @param {*} type_word
     * @description
     * Kiểm tra loại từ 
     * @returns 
     */
    #checkTypeWord(type_word)
    {
        let types = Object.keys(TYPE_WORD);
        for (let i = 0; i < types.length; i++)
        {
            if (type_word == types[i])
                return true;
        }
        return false;
    }

    /**
     * 
     * @param {*} word 
     * @param {*} meaning 
     * @param {*} type_word 
     * @description
     * Thêm một từ mới vào CSDL và tiến hành kiểm tra các điều kiện
     * @returns 
     */
    async onCreate(word, meaning, type_word = "")
    {
        let _infor = {  };
        let existed = Object.keys(await this.getWord(word))
            .length != 0;
        if (existed)
            throw new Error(`{} was existed!. If you want to update meaning please call update method!`);
        if (this.#checkTypeWord(type_word) == false) {
            type_word = TYPE_WORD.UNDEF;
            _infor["update_type"] = true;
        }
        // Giả thiết từ mới được tạo thì một ngày sau ôn!    
        _infor["result"] = await VocabularyModel.create({
            word, type : type_word, 
            meaning, forgot_rate : 1,
            timing : this.__user.timing + 1
        });
        return _infor;
    }
}

const vocabulary = new Vocabulary();
module.exports = vocabulary;