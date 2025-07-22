using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using System.Data.Entity;
using System.IO;
using System.Linq;
using System.Net;
using System.Net.Http;
using System.Threading.Tasks;
using System.Web;
using System.Web.Http;
using LifeinnovirorMentalHealthConsultency.Context;
using LifeinnovirorMentalHealthConsultency.Context.Tables;
using LifeinnovirorMentalHealthConsultency.Models;
using Newtonsoft.Json;

namespace LifeinnovirorMentalHealthConsultency.Controllers.PatientControllers
{
    public class PatientAccountManagementController : ApiController
    {
        private readonly LifeinnovirorContext db;    // Creating private db object to manupulate data
        public PatientAccountManagementController()
        {
            db = new LifeinnovirorContext(); // Initializing the database in constructor 
        }


        // it is an async function as it deals with files
        /*
         * Creates a patient with optional profile image.
         * Validates input, checks for duplicate email, saves data,
         * handles image upload with validation, logs action,
         * and returns success or error messages.
         */
        [HttpPost]
        [Route("api/patient/createPatient")]
        public async Task<IHttpActionResult> CreatePatient()
        {
            try
            {
                // Ensure request is multipart (required for file + JSON)
                if (!Request.Content.IsMimeMultipartContent())
                {
                    return Content(HttpStatusCode.BadRequest, new
                    {
                        success = false,
                        message = "Expected multipart content with patient data and optional profile image."
                    });
                }

                // Setup temp folder for file processing
                var tempUploadPath = HttpContext.Current.Server.MapPath("~/App_Data/Temp");
                if (!Directory.Exists(tempUploadPath))
                {
                    Directory.CreateDirectory(tempUploadPath);
                }

                // Read form-data (includes file + patient JSON)
                var provider = new MultipartFormDataStreamProvider(tempUploadPath);
                await Request.Content.ReadAsMultipartAsync(provider);

                // Extract JSON string
                var patientJson = provider.FormData["patient"];
                if (string.IsNullOrWhiteSpace(patientJson))
                {
                    return Content(HttpStatusCode.BadRequest, new
                    {
                        success = false,
                        message = "Missing patient data."
                    });
                }

                // Deserialize to Patient object
                var model = JsonConvert.DeserializeObject<Patient>(patientJson);
                if (model == null)
                {
                    return Content(HttpStatusCode.BadRequest, new
                    {
                        success = false,
                        message = "Invalid patient data format."
                    });
                }

                // Validate patient object manually
                var validationContext = new ValidationContext(model, null, null);
                var validationResults = new List<ValidationResult>();
                bool isValid = Validator.TryValidateObject(model, validationContext, validationResults, true);
                if (!isValid) //if not valid then send error message
                {
                    var errors = validationResults.Select(e => new
                    {
                        Field = string.Join(", ", e.MemberNames),
                        Errors = new List<string> { e.ErrorMessage }
                    });

                    return Content(HttpStatusCode.BadRequest, new
                    {
                        success = false,
                        message = "Validation failed.",
                        errors = errors
                    });
                }


                // Check for duplicate email
                bool emailExists = db.Patients.Any(p => p.Email == model.Email);
                if (emailExists)
                {
                    return Content(HttpStatusCode.Conflict, new
                    {
                        success = false,
                        message = "A patient with this email already exists."
                    });
                }

                // Hash password and set timestamps
                model.PasswordHash = CustomFunctions.CreateMD5(model.PasswordHash);
                model.CreatedAt = DateTime.Now;
                model.UpdatedAt = DateTime.Now;

                // Save patient first to get PatientId for image naming
                db.Patients.Add(model);
                await db.SaveChangesAsync();

                bool imageSaved = false; // Track if image save was successful
                string imageSaveErrorMessage = null; // To capture exact image upload error (if any)

                // Get uploaded image file (if any)
                var photo = provider.FileData.FirstOrDefault();
                if (photo != null)
                {
                    try
                    {
                        // Check file size limit (max 5 MB)
                        const int maxFileSizeInBytes = 5 * 1024 * 1024;
                        var fileInfo = new FileInfo(photo.LocalFileName);
                        if (fileInfo.Length > maxFileSizeInBytes)
                        {
                            File.Delete(photo.LocalFileName);
                            throw new Exception("Uploaded image must be less than 5 MB.");
                        }

                        // Validate file extension
                        var allowedExtensions = new[] { ".png", ".jpg", ".jpeg", ".heic" };
                        var extension = Path.GetExtension(photo.Headers.ContentDisposition.FileName.Trim('"')).ToLower();
                        if (!allowedExtensions.Contains(extension))
                        {
                            File.Delete(photo.LocalFileName);
                            throw new Exception("Only .png, .jpg, .jpeg, and .heic image formats are allowed.");
                        }

                        // Ensure target folder exists
                        var photoFolder = HttpContext.Current.Server.MapPath("~/App_Data/PatientProfilePhoto");
                        if (!Directory.Exists(photoFolder))
                        {
                            Directory.CreateDirectory(photoFolder);
                        }

                        // Prepare final path and move image
                        var finalFilePath = Path.Combine(photoFolder, model.PatientId + extension);
                        if (File.Exists(finalFilePath))
                        {
                            File.Delete(finalFilePath);
                        }

                        File.Move(photo.LocalFileName, finalFilePath);

                        // Update DB with image path
                        model.ProfilePhotoUrl = $"~/App_Data/PatientProfilePhoto/{model.PatientId}{extension}";
                        await db.SaveChangesAsync();
                        imageSaved = true; // Flag success
                    }
                    catch (Exception imgEx)
                    {
                        // Capture exact image error message to report later
                        imageSaveErrorMessage = imgEx.Message;
                    }
                }

                // Log patient creation action
                db.SystemLogs.Add(new SystemLog
                {
                    ActorType = "Patient",
                    ActorId = model.PatientId,
                    Action = "Create Patient",
                    Details = $"Patient '{model.Email}' created an account with id '{model.PatientId}'.",
                    CreatedAt = DateTime.Now
                });
                await db.SaveChangesAsync();

                // Determine final response message based on image status
                var message = photo == null // photo is null or not
                    ? "Patient created successfully." // No image uploaded
                    : imageSaved  //  image is saved or not
                        ? "Patient created successfully." // Image saved
                        : $"Patient created successfully. However, the profile image could not be saved: {imageSaveErrorMessage}"; // Image failed

                // Return response
                return Ok(new
                {
                    success = true,
                    message,
                    data = model
                });
            }
            catch (Exception ex)
            {
                // Unexpected server error
                return Content(HttpStatusCode.InternalServerError, new
                {
                    success = false,
                    message = "An unexpected error occurred while creating the patient.",
                    error = ex.Message
                });
            }
        }



        /// <summary>
        /// Updates an existing user account with new data and optional profile image.
        /// Validates input, replaces existing profile photo if provided,
        /// deletes previous photo file, and saves updated info to the database.
        /// Returns success or validation error response.
        /// </summary>
        [Authorize(Roles = "Patient")]
        [HttpPost]
        [Route("api/patient/updatePatient")]
        public async Task<IHttpActionResult> UpdatePatient()
        {
            try
            {
                // Ensure multipart content
                if (!Request.Content.IsMimeMultipartContent())
                {
                    return Content(HttpStatusCode.BadRequest, new
                    {
                        success = false,
                        message = "Expected multipart content with patient data and optional profile image."
                    });
                }

                var tempUploadPath = HttpContext.Current.Server.MapPath("~/App_Data/Temp");
                if (!Directory.Exists(tempUploadPath))
                {
                    Directory.CreateDirectory(tempUploadPath);
                }

                var provider = new MultipartFormDataStreamProvider(tempUploadPath);
                await Request.Content.ReadAsMultipartAsync(provider);

                var patientJson = provider.FormData["patient"];
                if (string.IsNullOrWhiteSpace(patientJson))
                {
                    return Content(HttpStatusCode.BadRequest, new
                    {
                        success = false,
                        message = "Missing patient data."
                    });
                }

                // Deserialize and get current patient
                var updatedModel = JsonConvert.DeserializeObject<Patient>(patientJson);
                if (updatedModel == null)
                {
                    return Content(HttpStatusCode.BadRequest, new
                    {
                        success = false,
                        message = "Invalid patient data format."
                    });
                }

                // manual validation
                var validationContext = new ValidationContext(updatedModel, null, null);
                var validationResults = new List<ValidationResult>();
                Validator.TryValidateObject(updatedModel, validationContext, validationResults, true);

                // Filter out PasswordHash errors
                var filteredResults = validationResults
                    .Where(r => !r.MemberNames.Contains("PasswordHash"))
                    .ToList();

                // if data is invalid then send invalid sms
                if (filteredResults.Any())
                {
                    var errors = filteredResults.Select(e => new
                    {
                        Field = string.Join(", ", e.MemberNames),
                        Errors = new List<string> { e.ErrorMessage }
                    });

                    return Content(HttpStatusCode.BadRequest, new
                    {
                        success = false,
                        message = "Validation failed.",
                        errors = errors
                    });
                }


                //check if logged in patient tries to change their data or others data
                // if tried to change others data then give unauthorized message
                if (CustomFunctions.GetPatientUserIdFromToken(User) != updatedModel.PatientId)
                {
                    return Content(HttpStatusCode.BadRequest, new
                    {
                        success = false,
                        message = "Unauthorize data Manupulation.",
                    });
                }

                // if data is valid then retrive patient id from it
                int patientId = updatedModel.PatientId;
                var patient = db.Patients.Find(patientId);
                if (patient == null)
                {
                    return Content(HttpStatusCode.NotFound, new
                    {
                        success = false,
                        message = "Patient not found."
                    });
                }

                // Check for email conflict excluding current object
                bool emailExists = db.Patients.Any(p => p.Email == updatedModel.Email &&
                                                   p.PatientId != patientId);
                if (emailExists)
                {
                    return Content(HttpStatusCode.Conflict, new
                    {
                        success = false,
                        message = "Another account with this email already exists."
                    });
                }

                // Update patient fields
                patient.FullName = updatedModel.FullName;
                patient.DateOfBirth = updatedModel.DateOfBirth;
                patient.Gender = updatedModel.Gender;
                patient.Email = updatedModel.Email;
                patient.PhoneNumber = updatedModel.PhoneNumber;
                patient.Address = updatedModel.Address;
                patient.EmergencyNumber = updatedModel.EmergencyNumber;
                patient.MedicalHistoryText = updatedModel.MedicalHistoryText;
                patient.Height = updatedModel.Height;
                patient.Weight = updatedModel.Weight;
                patient.Religion = updatedModel.Religion;
                patient.EducationDetails = updatedModel.EducationDetails;
                patient.Allergies = updatedModel.Allergies;
                patient.SkinTone = updatedModel.SkinTone;
                patient.UpdatedAt = DateTime.Now;

                // Handle profile image (optional)
                bool imageSaved = false; // Track if image save was successful
                string imageSaveErrorMessage = null; // To capture exact image upload error (if any)

                // Get uploaded image file (if any)
                var photo = provider.FileData.FirstOrDefault();
                if (photo != null)
                {
                    try
                    {
                        // Check file size limit (max 5 MB)
                        const int maxFileSizeInBytes = 5 * 1024 * 1024;
                        var fileInfo = new FileInfo(photo.LocalFileName);
                        if (fileInfo.Length > maxFileSizeInBytes)
                        {
                            File.Delete(photo.LocalFileName);
                            throw new Exception("Uploaded image must be less than 5 MB.");
                        }

                        // Validate file extension
                        var allowedExtensions = new[] { ".png", ".jpg", ".jpeg", ".heic" };
                        var extension = Path.GetExtension(photo.Headers.ContentDisposition.FileName.Trim('"')).ToLower();
                        if (!allowedExtensions.Contains(extension))
                        {
                            File.Delete(photo.LocalFileName);
                            throw new Exception("Only .png, .jpg, .jpeg, and .heic image formats are allowed.");
                        }

                        // Ensure target folder exists
                        var photoFolder = HttpContext.Current.Server.MapPath("~/App_Data/PatientProfilePhoto");
                        if (!Directory.Exists(photoFolder))
                        {
                            Directory.CreateDirectory(photoFolder);
                        }

                        // CLEANUP: Delete any previous image for this patient
                        var existingFiles = Directory.GetFiles(photoFolder, $"{patientId}.*");
                        foreach (var file in existingFiles) 
                        {
                            // it will delete all image even with different extension file for that user
                            File.Delete(file); 
                        }

                        // Prepare final path and move image
                        var finalFilePath = Path.Combine(photoFolder, patientId + extension);
                        File.Move(photo.LocalFileName, finalFilePath);

                        // Update DB with image path
                        patient.ProfilePhotoUrl = $"~/App_Data/PatientProfilePhoto/{patient.PatientId}{extension}";
                        await db.SaveChangesAsync();
                        imageSaved = true; // Flag success
                    }
                    catch (Exception imgEx)
                    {
                        // Capture exact image error message to report later
                        imageSaveErrorMessage = imgEx.Message;
                    }
                }

                // Log patient creation action
                db.SystemLogs.Add(new SystemLog
                {
                    ActorType = "Patient",
                    ActorId = patientId,
                    Action = "Update Patient",
                    Details = $"Patient '{updatedModel.Email}' updated their account, id '{updatedModel.PatientId}'.",
                    CreatedAt = DateTime.Now
                });
                await db.SaveChangesAsync();

                // Determine final response message based on image status
                var message = photo == null // photo is null or not
                    ? "Patient updated successfully." // No image uploaded
                    : imageSaved  //  image is saved or not
                        ? "Patient updated successfully." // Image saved
                        : $"Patient updated successfully. However, the profile image could not be saved: {imageSaveErrorMessage}"; // Image failed

                // Return response
                return Ok(new
                {
                    success = true,
                    message,
                    data = patient
                });
            }
            catch (Exception ex)
            {
                // Unexpected server error
                return Content(HttpStatusCode.InternalServerError, new
                {
                    success = false,
                    message = "An unexpected error occurred while updating the patient.",
                    error = ex.Message
                });
            }
        }





        // it retrives specific patient data with patient image
        // it send the image data which can be render by below code
        // <img src="data:image/jpeg;base64,{ProfilePhotoBase64}" />
        [Authorize(Roles = "Patient")]
        [HttpGet]
        [Route("api/patient/getPatient/{id:int}")]
        public IHttpActionResult GetPatient(int id)
        {
            try
            {
                var patient = db.Patients.Find(id);
                // if not found 
                if (patient == null)
                {
                    return Content(HttpStatusCode.NotFound, new
                    {
                        success = false,
                        message = "Patient not found."
                    });
                }

                // maping profile image path to retrive it
                string photoPath = HttpContext.Current.Server.MapPath(patient.ProfilePhotoUrl ?? "");
                string base64Image = null;

                //send as base64 data
                if (!string.IsNullOrEmpty(patient.ProfilePhotoUrl) && File.Exists(photoPath))
                {
                    byte[] imageBytes = File.ReadAllBytes(photoPath);
                    base64Image = Convert.ToBase64String(imageBytes);
                }

                return Ok(new
                {
                    success = true,
                    message = "Patient retrieved successfully.",
                    data = new
                    {
                        patient.PatientId,
                        patient.FullName,
                        patient.DateOfBirth,
                        patient.Gender,
                        patient.Email,
                        patient.PhoneNumber,
                        patient.Address,
                        patient.EmergencyNumber,
                        patient.MedicalHistoryText,
                        patient.Height,
                        patient.Weight,
                        patient.Religion,
                        patient.EducationDetails,
                        patient.Allergies,
                        patient.SkinTone,
                        patient.CreatedAt,
                        patient.UpdatedAt,
                        ProfilePhotoBase64 = base64Image
                    }
                });
            }
            catch (Exception ex)
            {
                // Unexpected server error
                return Content(HttpStatusCode.InternalServerError, new
                {
                    success = false,
                    message = "An unexpected error occurred while retriving the patient.",
                    error = ex.Message
                });
            }
        }



        /// <summary>
        /// Deletes the currently logged-in patient account along with the profile image (if any).
        /// </summary>
        /// <returns>Success message or error response.</returns>
        [Authorize(Roles = "Patient")]
        [HttpDelete]
        [Route("api/patient/deleteAccount")]
        public async Task<IHttpActionResult> DeleteAccount()
        {
            try
            {
                // Get current patient ID from token
                var patientId = CustomFunctions.GetPatientUserIdFromToken(User);

                // Find patient in the database
                var patient = await db.Patients.FindAsync(patientId);
                if (patient == null)
                {
                    return Content(HttpStatusCode.NotFound, new
                    {
                        success = false,
                        message = "Patient not found."
                    });
                }

                // Delete profile photo if exists
                var photoFolder = HttpContext.Current.Server.MapPath("~/App_Data/PatientProfilePhoto");
                var existingFiles = Directory.GetFiles(photoFolder, $"{patientId}.*");
                foreach (var file in existingFiles)
                {
                    File.Delete(file);
                }

                // Delete patient record
                db.Patients.Remove(patient);
                await db.SaveChangesAsync();

                // Log patient deletion action
                db.SystemLogs.Add(new SystemLog
                {
                    ActorType = "Patient",
                    ActorId = patientId,
                    Action = "Delete Patient",
                    Details = $"Patient '{patient.Email}' deleted their account, id '{patient.PatientId}'.",
                    CreatedAt = DateTime.Now
                });
                await db.SaveChangesAsync();
                return Ok(new
                {
                    success = true,
                    message = "Account deleted successfully."
                });
            }
            catch (Exception ex)
            {
                // Unexpected server error
                return Content(HttpStatusCode.InternalServerError, new
                {
                    success = false,
                    message = "An unexpected error occurred while deleting the patient Account.",
                    error = ex.Message
                });
            }
        }




        [Authorize(Roles = "Patient")]
        [HttpPost]
        [Route("api/patient/changePassword")]
        public async Task<IHttpActionResult> ChangePassword(ChangePasswordModel model)
        {
            try
            {
                // Model validation
                if (!ModelState.IsValid)
                {
                    var errors = ModelState.Where(ms => ms.Value.Errors.Any())
                        .Select(ms => new
                        {
                            Field = ms.Key,
                            Errors = ms.Value.Errors.Select(e => e.ErrorMessage).ToList()
                        });

                    return Content(HttpStatusCode.BadRequest, new
                    {
                        success = false,
                        message = "Validation failed.",
                        errors = errors
                    });
                }


                // Verify if current user exists or not 
                var patient = await db.Patients.FindAsync(CustomFunctions.GetPatientUserIdFromToken(User));
                if (patient == null)
                {
                    return Content(HttpStatusCode.NotFound, new
                    {
                        success = false,
                        message = "Patient not found."
                    });
                }

                //verify current password
                if (patient.PasswordHash != CustomFunctions.CreateMD5(model.CurrentPassword))
                {
                    return Content(HttpStatusCode.BadRequest, new
                    {
                        success = false,
                        message = "Current password is incorrect."
                    });
                }

                // Update password
                patient.PasswordHash = CustomFunctions.CreateMD5(model.NewPassword);
                await db.SaveChangesAsync();

                return Ok(new
                {
                    success = true,
                    message = "Password changed successfully."
                });
            }
            catch (Exception ex)
            {
                // Unexpected server error
                return Content(HttpStatusCode.InternalServerError, new
                {
                    success = false,
                    message = "An unexpected error occurred while changing password.",
                    error = ex.Message
                });
            }

        }

    }
}
