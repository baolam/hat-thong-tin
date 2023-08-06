const bcrypt = require("bcrypt");
const shortid = require("shortid");

const Database = require("../config/database");
const sequelize = require("sequelize");

const UserModel = Database.define(
    process.env.DB_USER_TABLE, {
    id : {
        type : sequelize.STRING,
        allowNull : false,
        primaryKey : true
    },
    name : {
        type : sequelize.TEXT,
        allowNull : false
    },
    mail : {
        type : sequelize.STRING,
        allowNull : false
    },
    password : {
        type : sequelize.TEXT,
        allowNull : false
    },
    avatar : {
        type : sequelize.TEXT,
        comment : "Đường dẫn tới avatar người dùng"
    }
});

class User {
    async initalize()
    {
        return UserModel.sync();
    }

    async onGetUser(mail)
    {
        return UserModel.findOne({
            where : { mail }
        }).then(vl => vl.toJSON());
    }

    /**
     * 
     * @param {*} name 
     * @param {*} mail 
     * @param {*} password 
     * @param {*} avatar 
     * @description
     * Tạo một người dùng mới
     * @returns 
     */
    async onCreate(name, mail, password, avatar)
    {
        let hash_password = await bcrypt.hash(password, 11);
        /// Cần tối ưu chỗ ni
        /// Hai lần request
        let _user = (await UserModel.findOne({ where : { mail } }))
            .toJSON();
        if (Object.keys(_user).length != 0)
            throw new Error(`User with mail ${mail} existed!`);
        return UserModel.create({
            name, mail,
            id : shortid.generate(),
            password : hash_password,
            avatar
        });
    }

    /**
     * 
     * @param {*} mail 
     * @param {*} password 
     * @param {*} wanna_update 
     * @description
     * Các tham số muốn cập nhật thì dựa vào wanna_update
     * @returns 
     */
    async onUpdateBasedMail(mail, password, wanna_update = {})
    {
        let user = await this.onGetUser(mail);
        if (Object.keys(user).length === 0)
            throw new Error(`User with ${mail} does not exist!`);
        let state = await bcrypt.compare(password, user.password);
        if (state == false)
            throw new Error(`Not permission for updating!`);
        return UserModel.update(wanna_update, { where : { mail } });
    }

    async onUpdateBasedMail(mail, wanna_update = {})
    {

    }

    /**
     * 
     * @param {*} mail 
     * @param {*} password
     * @description
     * Đăng nhập 
     * @returns 
     */
    async logIn(mail, password)
    {
        let user = await this.onGetUser(mail);
        if (Object.keys(user).length === 0)
            throw new Error(`User with ${mail} does not exist!`);
        return [await bcrypt.compare(password, user.password), user.id];
    }
}

const user = new User();
module.exports = user;