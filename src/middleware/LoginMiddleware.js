module.exports = (req, res, next) => {
    if (req.signedCookies == undefined || req.signedCookies.user_id == undefined)
        res.send({ login : false });
    else next();
}