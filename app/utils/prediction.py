from sqlalchemy.orm import Session
from app.models.who_standards import HeightForAge, WeightForAge
from app.models.child import Child


def get_height_for_age_data(db: Session, age_years: int, gender: str):
    return db.query(HeightForAge).filter(
        HeightForAge.year == age_years,
        HeightForAge.gender == gender
    ).first()


def get_weight_for_age_data(db: Session, age_years: int, gender: str):
    return db.query(WeightForAge).filter(
        WeightForAge.year == age_years,
        WeightForAge.gender == gender
    ).first()


def predict_stunting(db: Session, child: Child):
    age_months = child.age
    height_for_age = get_height_for_age_data(db, age_months, child.gender)

    if height_for_age:
        if child.height < height_for_age.sd_2_negative:
            return True
    return False


def predict_wasting(db: Session, child: Child):
    age_months = child.age
    weight_for_age = get_weight_for_age_data(db, age_months, child.gender)

    if weight_for_age:
        if child.weight < weight_for_age.sd_2_negative:
            return True
    return False


def predict_overweight(db: Session, child: Child):
    age_months = child.age
    weight_for_age = get_weight_for_age_data(db, age_months, child.gender)

    if weight_for_age:
        if child.weight > weight_for_age.sd_2_positive:
            return True
    return False


def predict_child_condition(db: Session, child):
    is_stunting = predict_stunting(db, child)
    is_wasting = predict_wasting(db, child)
    is_overweight = predict_overweight(db, child)

    return {
        "is_stunting": is_stunting,
        "is_wasting": is_wasting,
        "is_overweight": is_overweight
    }

