/*
var hang = categorical_features['hang']
var dong_xe = categorical_features['dong_xe']
var xuat_xu = categorical_features['xuat_xu']
var kieu_dang = categorical_features['kieu_dang']
var nhien_lieu = categorical_features['nhien_lieu']
var hop_so = categorical_features['hop_so']
*/
function add_tinh_trang(divName){
    console.log('Hello')
    var categorical_features = JSON.parse(categorical_features);
    var tinh_trang = categorical_features['tinh_trang']
    console.log(tinh_trang)
    var selectHTML = "";
    for(i in tinh_trang){
        selectHTML += `<option value=${tinh_trang[i]}>${tinh_trang[i]}</option>`;
    }
    document.getElementById(divName).innerHTML = selectHTML;
}
//var categorical_features = require('D:/20211/KHDL/KHDL_IT4930/inference/misc/categorical_features.json');
//console.log(categorical_features)
$(document).ready(function(){
    load_data();
    function load_data(query = ''){
        $ajax({
            url : "/all_features",
            method:"GET",
            data
        })
    }
})